"""Tests for document upload and management functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from bankscope.chat.store import ChatStore


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    """Create a temporary ChatStore for testing."""
    chat_store = ChatStore(tmp_path / "chat.db")
    chat_store.initialize()
    return chat_store


class TestDocumentUpload:
    """Test document upload functionality."""

    def test_upload_text_document(self, store: ChatStore) -> None:
        """Test uploading a plain text document."""
        content = b"This is a test document.\nIt has multiple lines.\nAnd some content."
        doc = store.upload_document(
            thread_id=None,
            filename="test.txt",
            content_type="text/plain",
            file_content=content,
            metadata={"test": "value"},
        )

        assert doc["id"] is not None
        assert doc["filename"] == "test.txt"
        assert doc["content_type"] == "text/plain"
        assert doc["file_size"] == len(content)
        assert doc["parsed"] is True
        assert doc["parse_error"] is None
        assert doc["metadata"]["test"] == "value"

    def test_upload_markdown_document(self, store: ChatStore) -> None:
        """Test uploading a Markdown document."""
        content = b"# Test Document\n\nThis is **bold** text."
        doc = store.upload_document(
            thread_id=None,
            filename="test.md",
            content_type="text/markdown",
            file_content=content,
        )

        assert doc["filename"] == "test.md"
        assert doc["content_type"] == "text/markdown"
        assert doc["parsed"] is True

    def test_upload_csv_document(self, store: ChatStore) -> None:
        """Test uploading a CSV document."""
        content = b"col1,col2,col3\n1,2,3\n4,5,6"
        doc = store.upload_document(
            thread_id=None,
            filename="test.csv",
            content_type="text/csv",
            file_content=content,
        )

        assert doc["filename"] == "test.csv"
        assert doc["content_type"] == "text/csv"
        assert doc["parsed"] is True

    def test_upload_json_document(self, store: ChatStore) -> None:
        """Test uploading a JSON document."""
        content = b'{"key": "value", "array": [1, 2, 3]}'
        doc = store.upload_document(
            thread_id=None,
            filename="test.json",
            content_type="application/json",
            file_content=content,
        )

        assert doc["filename"] == "test.json"
        assert doc["content_type"] == "application/json"
        assert doc["parsed"] is True

    def test_upload_empty_document_fails(self, store: ChatStore) -> None:
        """Test that uploading an empty document raises ValueError."""
        with pytest.raises(ValueError, match="Document is empty"):
            store.upload_document(
                thread_id=None,
                filename="empty.txt",
                content_type="text/plain",
                file_content=b"",
            )

    def test_upload_oversized_document_fails(self, store: ChatStore) -> None:
        """Test that uploading a document exceeding size limit raises ValueError."""
        # 10MB + 1 byte
        content = b"x" * (10 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="File size exceeds maximum"):
            store.upload_document(
                thread_id=None,
                filename="huge.txt",
                content_type="text/plain",
                file_content=content,
            )

    def test_upload_with_inferred_content_type(self, store: ChatStore) -> None:
        """Test uploading a document with inferred content type from filename."""
        content = b"test content"
        # Empty content_type should be inferred from filename
        doc = store.upload_document(
            thread_id=None,
            filename="test.md",
            content_type="",
            file_content=content,
        )

        assert doc["content_type"] == "text/markdown"

    def test_upload_with_thread_id(self, store: ChatStore) -> None:
        """Test uploading a document associated with a thread."""
        thread_id = store.create_thread("Document thread")["id"]
        content = b"thread document content"
        doc = store.upload_document(
            thread_id=thread_id,
            filename="thread_doc.txt",
            content_type="text/plain",
            file_content=content,
        )

        assert doc["thread_id"] == thread_id


class TestDocumentRetrieval:
    """Test document retrieval functionality."""

    def test_list_documents(self, store: ChatStore) -> None:
        """Test listing documents."""
        # Upload multiple documents
        for i in range(3):
            store.upload_document(
                thread_id=None,
                filename=f"doc{i}.txt",
                content_type="text/plain",
                file_content=f"content {i}".encode(),
            )

        docs = store.list_documents()
        assert len(docs) == 3

        # Check that documents are ordered by uploaded_at descending
        filenames = [doc["filename"] for doc in docs]
        assert filenames == ["doc2.txt", "doc1.txt", "doc0.txt"]

    def test_list_documents_by_thread(self, store: ChatStore) -> None:
        """Test listing documents filtered by thread_id."""
        thread1 = store.create_thread("Thread one")["id"]
        thread2 = store.create_thread("Thread two")["id"]

        # Upload documents to different threads
        store.upload_document(
            thread_id=thread1,
            filename="thread1_doc.txt",
            content_type="text/plain",
            file_content=b"thread1 content",
        )
        store.upload_document(
            thread_id=thread2,
            filename="thread2_doc.txt",
            content_type="text/plain",
            file_content=b"thread2 content",
        )
        store.upload_document(
            thread_id=None,
            filename="no_thread_doc.txt",
            content_type="text/plain",
            file_content=b"no thread content",
        )

        # List documents for thread1
        docs = store.list_documents(thread_id=thread1)
        assert len(docs) == 1
        assert docs[0]["filename"] == "thread1_doc.txt"

        # List documents for thread2
        docs = store.list_documents(thread_id=thread2)
        assert len(docs) == 1
        assert docs[0]["filename"] == "thread2_doc.txt"

    def test_get_document(self, store: ChatStore) -> None:
        """Test getting a single document by ID."""
        uploaded = store.upload_document(
            thread_id=None,
            filename="test.txt",
            content_type="text/plain",
            file_content=b"test content",
        )

        doc = store.get_document(uploaded["id"])
        assert doc["id"] == uploaded["id"]
        assert doc["filename"] == "test.txt"
        assert doc["content_type"] == "text/plain"

    def test_get_document_not_found(self, store: ChatStore) -> None:
        """Test that getting a non-existent document raises KeyError."""
        with pytest.raises(KeyError):
            store.get_document("non-existent-id")

    def test_get_document_content(self, store: ChatStore) -> None:
        """Test getting document raw content."""
        content = b"raw document content"
        uploaded = store.upload_document(
            thread_id=None,
            filename="test.txt",
            content_type="text/plain",
            file_content=content,
        )

        retrieved_content = store.get_document_content(uploaded["id"])
        assert retrieved_content == content

    def test_get_document_content_not_found(self, store: ChatStore) -> None:
        """Test that getting content of non-existent document raises KeyError."""
        with pytest.raises(KeyError):
            store.get_document_content("non-existent-id")

    def test_get_document_text(self, store: ChatStore) -> None:
        """Test getting parsed document text."""
        content = b"Parsed text content\nwith multiple lines"
        uploaded = store.upload_document(
            thread_id=None,
            filename="test.txt",
            content_type="text/plain",
            file_content=content,
        )

        text = store.get_document_text(uploaded["id"])
        assert text == "Parsed text content\nwith multiple lines"

    def test_get_document_text_not_found(self, store: ChatStore) -> None:
        """Test that getting text of non-existent document raises KeyError."""
        with pytest.raises(KeyError):
            store.get_document_text("non-existent-id")


class TestDocumentDeletion:
    """Test document deletion functionality."""

    def test_delete_document(self, store: ChatStore) -> None:
        """Test deleting a document."""
        uploaded = store.upload_document(
            thread_id=None,
            filename="test.txt",
            content_type="text/plain",
            file_content=b"test content",
        )

        store.delete_document(uploaded["id"])

        # Verify document is deleted
        with pytest.raises(KeyError):
            store.get_document(uploaded["id"])

    def test_delete_document_not_found(self, store: ChatStore) -> None:
        """Test that deleting a non-existent document raises KeyError."""
        with pytest.raises(KeyError):
            store.delete_document("non-existent-id")

    def test_delete_document_updates_list(self, store: ChatStore) -> None:
        """Test that deleting a document updates the document list."""
        # Upload 3 documents
        doc_ids = []
        for i in range(3):
            doc = store.upload_document(
                thread_id=None,
                filename=f"doc{i}.txt",
                content_type="text/plain",
                file_content=f"content {i}".encode(),
            )
            doc_ids.append(doc["id"])

        # Verify all documents exist
        assert len(store.list_documents()) == 3

        # Delete middle document
        store.delete_document(doc_ids[1])

        # Verify only 2 documents remain
        assert len(store.list_documents()) == 2

        # Verify the remaining documents are the expected ones
        remaining_ids = [doc["id"] for doc in store.list_documents()]
        assert doc_ids[0] in remaining_ids
        assert doc_ids[1] not in remaining_ids
        assert doc_ids[2] in remaining_ids


class TestDocumentParsing:
    """Test document parsing functionality."""

    def test_parse_text_document(self, store: ChatStore) -> None:
        """Test parsing of text documents."""
        content = b"Simple text content"
        parsed = store._parse_document(content, "text/plain", "test.txt")
        assert parsed == "Simple text content"

    def test_parse_markdown_document(self, store: ChatStore) -> None:
        """Test parsing of Markdown documents."""
        content = b"# Heading\n\nParagraph text"
        parsed = store._parse_document(content, "text/markdown", "test.md")
        assert "# Heading" in parsed
        assert "Paragraph text" in parsed

    def test_parse_json_document(self, store: ChatStore) -> None:
        """Test parsing of JSON documents."""
        content = b'{"key": "value"}'
        parsed = store._parse_document(content, "application/json", "test.json")
        assert parsed == '{"key": "value"}'

    def test_parse_csv_document(self, store: ChatStore) -> None:
        """Test parsing of CSV documents."""
        content = b"col1,col2\nval1,val2"
        parsed = store._parse_document(content, "text/csv", "test.csv")
        assert "col1,col2" in parsed
        assert "val1,val2" in parsed

    def test_parse_large_document_truncation(self, store: ChatStore) -> None:
        """Test that large documents are truncated."""
        # Create content larger than _MAX_PARSED_DOCUMENT_CHARS (200,000)
        content = b"x" * 300_000
        parsed = store._parse_document(content, "text/plain", "large.txt")

        # Should be truncated
        assert len(parsed) < 300_000
        assert "[Document truncated at the upload context limit.]" in parsed

    def test_parse_empty_document_fails(self, store: ChatStore) -> None:
        """Test that parsing an empty document raises ValueError."""
        with pytest.raises(ValueError, match="no extractable text"):
            store._parse_document(b"", "text/plain", "empty.txt")

    def test_parse_whitespace_only_document_fails(self, store: ChatStore) -> None:
        """Test that parsing a whitespace-only document raises ValueError."""
        with pytest.raises(ValueError, match="no extractable text"):
            store._parse_document(b"   \n\t  ", "text/plain", "whitespace.txt")


class TestDocumentInference:
    """Test content type inference from filename."""

    def test_infer_content_type_txt(self, store: ChatStore) -> None:
        """Test inferring content type from .txt extension."""
        assert store._infer_content_type("test.txt") == "text/plain"

    def test_infer_content_type_md(self, store: ChatStore) -> None:
        """Test inferring content type from .md extension."""
        assert store._infer_content_type("test.md") == "text/markdown"

    def test_infer_content_type_pdf(self, store: ChatStore) -> None:
        """Test inferring content type from .pdf extension."""
        assert store._infer_content_type("test.pdf") == "application/pdf"

    def test_infer_content_type_csv(self, store: ChatStore) -> None:
        """Test inferring content type from .csv extension."""
        assert store._infer_content_type("test.csv") == "text/csv"

    def test_infer_content_type_docx(self, store: ChatStore) -> None:
        """Test inferring content type from .docx extension."""
        assert (
            store._infer_content_type("test.docx")
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_infer_content_type_unknown(self, store: ChatStore) -> None:
        """Test that unknown extensions return None."""
        assert store._infer_content_type("test.xyz") is None

    def test_infer_content_type_case_insensitive(self, store: ChatStore) -> None:
        """Test that content type inference is case-insensitive."""
        assert store._infer_content_type("test.TXT") == "text/plain"
        assert store._infer_content_type("test.PDF") == "application/pdf"
