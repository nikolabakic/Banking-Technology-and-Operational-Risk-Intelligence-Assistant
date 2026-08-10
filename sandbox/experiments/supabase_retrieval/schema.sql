create extension if not exists vector;

create schema if not exists bankscope_supabase_experiment;

create table if not exists bankscope_supabase_experiment.documents (
    ordinal integer primary key,
    record_id text not null unique,
    target_chunk_id text not null,
    record_type text not null check (record_type in ('text', 'table')),
    ticker text not null,
    bank_name text not null,
    embedding_text text not null,
    metadata jsonb not null,
    embedding vector(1024) not null,
    fts tsvector generated always as (
        to_tsvector('english', embedding_text)
    ) stored
);

create index if not exists documents_embedding_hnsw
on bankscope_supabase_experiment.documents
using hnsw (embedding vector_cosine_ops);

create index if not exists documents_fts_gin
on bankscope_supabase_experiment.documents
using gin (fts);

create index if not exists documents_filter_btree
on bankscope_supabase_experiment.documents (ticker, record_type);

