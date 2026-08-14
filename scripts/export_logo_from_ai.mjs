import { inflateSync } from "node:zlib";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const projectRoot = resolve(import.meta.dirname, "..");
const brandRoot = resolve(projectRoot, "assets", "brand");
const sourcePath = resolve(brandRoot, "bankscope.ai");
const source = readFileSync(sourcePath);
const latin = source.toString("latin1");

function inflateObject(objectId) {
  const objectStart = latin.indexOf(`${objectId} 0 obj`);
  if (objectStart < 0) throw new Error(`PDF object ${objectId} was not found.`);
  let streamStart = latin.indexOf("stream", objectStart) + "stream".length;
  if (latin.slice(streamStart, streamStart + 2) === "\r\n") streamStart += 2;
  else if (latin[streamStart] === "\r" || latin[streamStart] === "\n") streamStart += 1;
  const streamEnd = latin.indexOf("endstream", streamStart);
  return inflateSync(source.subarray(streamStart, streamEnd)).toString("latin1");
}

const brandColors = {
  blue: "#3459b1",
  red: "#ee413b",
  orange: "#f36b2d",
  white: "#ffffff",
};

function resolveColor(values) {
  const key = values.map((value) => Number(value).toFixed(2)).join(",");
  const colors = {
    "0.00,0.00,0.00,0.00": brandColors.white,
    "0.00,0.90,0.85,0.00": brandColors.red,
    "0.00,0.80,0.95,0.00": brandColors.orange,
    "1.00,0.90,0.00,0.00": brandColors.blue,
    "1.00,1.00,0.00,0.00": brandColors.blue,
  };
  return colors[key] ?? brandColors.blue;
}

function transformPoint(matrix, x, y) {
  const [a, b, c, d, e, f] = matrix;
  return [a * x + c * y + e, b * x + d * y + f];
}

function multiply(left, right) {
  const [a, b, c, d, e, f] = left;
  const [g, h, i, j, k, l] = right;
  return [
    a * g + c * h,
    b * g + d * h,
    a * i + c * j,
    b * i + d * j,
    a * k + c * l + e,
    b * k + d * l + f,
  ];
}

function parseArtwork(content) {
  const elements = [];
  const stack = [];
  let matrix = [1, 0, 0, 1, 0, 0];
  let fill = brandColors.blue;
  let stroke = brandColors.blue;
  let strokeWidth = 1;
  let path = [];
  let points = [];

  const resetPath = () => {
    path = [];
    points = [];
  };

  const addPoints = (coordinates) => {
    for (let index = 0; index < coordinates.length; index += 2) {
      points.push(transformPoint(matrix, coordinates[index], coordinates[index + 1]));
    }
  };

  const emit = (kind) => {
    if (!path.length || !points.length) return resetPath();
    const xs = points.map(([x]) => x);
    const ys = points.map(([, y]) => y);
    elements.push({
      d: path.join(" "),
      matrix: [...matrix],
      fill: kind === "fill" ? fill : "none",
      stroke: kind === "stroke" ? stroke : "none",
      strokeWidth,
      bounds: {
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys),
      },
    });
    resetPath();
  };

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line === "EMC") continue;
    const tokens = line.split(/\s+/);
    const operator = tokens.at(-1);
    const values = tokens.slice(0, -1).map(Number).filter(Number.isFinite);

    if (tokens[0] === "q" && operator === "cm") {
      stack.push({ matrix: [...matrix], fill, stroke, strokeWidth });
      matrix = multiply(matrix, values);
    } else if (operator === "q") {
      stack.push({ matrix: [...matrix], fill, stroke, strokeWidth });
    } else if (operator === "Q") {
      const state = stack.pop();
      if (state) ({ matrix, fill, stroke, strokeWidth } = state);
      resetPath();
    } else if (operator === "cm") {
      matrix = multiply(matrix, values);
    } else if (operator === "scn") {
      fill = resolveColor(values);
    } else if (operator === "SCN") {
      stroke = resolveColor(values);
    } else if (operator === "w") {
      strokeWidth = values[0];
    } else if (operator === "m") {
      path.push(`M${values[0]} ${values[1]}`);
      addPoints(values);
    } else if (operator === "l") {
      path.push(`L${values[0]} ${values[1]}`);
      addPoints(values);
    } else if (operator === "c") {
      path.push(`C${values.join(" ")}`);
      addPoints(values);
    } else if (operator === "re") {
      const [x, y, width, height] = values;
      path.push(`M${x} ${y}h${width}v${height}h${-width}Z`);
      addPoints([x, y, x + width, y + height]);
    } else if (operator === "h") {
      path.push("Z");
    } else if (operator === "f") {
      emit("fill");
    } else if (operator === "S") {
      emit("stroke");
    } else if (operator === "n") {
      resetPath();
    }
  }

  return elements.filter(({ bounds }) => !(bounds.minX < 2 && bounds.maxX > 1100 && bounds.maxY > 800));
}

function svg(elements, crop, label) {
  const paths = elements.map((element) => {
    const matrix = ` transform="matrix(${element.matrix.join(" ")})"`;
    const stroke = element.stroke === "none"
      ? ""
      : ` stroke="${element.stroke}" stroke-width="${element.strokeWidth}"`;
    return `<path d="${element.d}" fill="${element.fill}"${stroke}${matrix}/>`;
  }).join("");
  const [left, top, width, height, pdfTop] = crop;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}"><g transform="matrix(1 0 0 -1 ${-left} ${pdfTop})">${paths}</g></svg>\n`;
}

const artwork = parseArtwork(inflateObject(33));
const markArtwork = artwork.filter(({ bounds }) => (
  bounds.minX >= 165 && bounds.maxX <= 305 && bounds.minY >= 495 && bounds.maxY <= 705
));
const targetArtwork = artwork
  .filter(({ bounds }) => (
    bounds.minX >= 465 && bounds.maxX <= 720 && bounds.minY >= 215 && bounds.maxY <= 470
  ))
  .map((element) => ({
    ...element,
    fill: element.fill === brandColors.white ? brandColors.blue : element.fill,
    stroke: element.stroke === brandColors.white ? brandColors.blue : element.stroke,
  }));

const outputs = [
  [resolve(brandRoot, "bankscope-wordmark.svg"), svg(artwork, [130, 140, 880, 500, 700], "BankScope")],
  [resolve(brandRoot, "bankscope-mark.svg"), svg(markArtwork, [165, 140, 140, 210, 700], "BankScope")],
  [resolve(brandRoot, "bankscope-target.svg"), svg(targetArtwork, [465, 230, 255, 255, 470], "BankScope target")],
];

for (const [rootPath, contents] of outputs) {
  const publicPath = resolve(projectRoot, "frontend", "public", "brand", rootPath.split(/[\\/]/).at(-1));
  mkdirSync(dirname(publicPath), { recursive: true });
  writeFileSync(rootPath, contents);
  writeFileSync(publicPath, contents);
  console.log(`Wrote ${rootPath}`);
  console.log(`Wrote ${publicPath}`);
}
