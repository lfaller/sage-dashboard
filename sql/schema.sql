-- SAGE Dashboard Database Schema
-- PostgreSQL schema for Supabase

-- Studies table: GEO/SRA study information
CREATE TABLE IF NOT EXISTS studies (
    id SERIAL PRIMARY KEY,
    geo_accession VARCHAR(20) UNIQUE,
    sra_accession VARCHAR(20),
    title TEXT,
    summary TEXT,
    organism VARCHAR(100),
    platform VARCHAR(50),
    study_type VARCHAR(50),
    publication_date DATE,
    pubmed_id VARCHAR(20),
    sample_count INTEGER,

    -- Metadata quality scores
    has_sex_metadata BOOLEAN DEFAULT FALSE,
    sex_metadata_completeness FLOAT DEFAULT 0.0,
    sex_inferrable BOOLEAN DEFAULT FALSE,
    sex_inference_confidence FLOAT,

    -- Analysis status
    reports_sex_analysis BOOLEAN DEFAULT FALSE,
    reports_sex_difference BOOLEAN,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Samples table: Individual sample metadata
CREATE TABLE IF NOT EXISTS samples (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    geo_accession VARCHAR(20),
    sra_accession VARCHAR(20),

    -- Reported metadata
    reported_sex VARCHAR(20),
    reported_age VARCHAR(50),
    reported_tissue VARCHAR(100),
    reported_disease VARCHAR(200),

    -- Inferred metadata
    inferred_sex VARCHAR(20),
    inference_method VARCHAR(50),
    inference_confidence FLOAT,

    -- Flags
    sex_mismatch BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Disease mappings table
CREATE TABLE IF NOT EXISTS disease_mappings (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES studies(id) ON DELETE CASCADE,
    disease_term VARCHAR(200),
    doid_id VARCHAR(20),
    doid_name VARCHAR(200),
    mesh_id VARCHAR(20),
    disease_category VARCHAR(100),

    -- Clinical relevance
    known_sex_difference BOOLEAN DEFAULT FALSE,
    sex_bias_direction VARCHAR(20),
    clinical_priority_score FLOAT
);

-- Historical snapshots for tracking progress
CREATE TABLE IF NOT EXISTS completeness_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE,
    disease_category VARCHAR(100),
    organism VARCHAR(100),

    total_studies INTEGER,
    studies_with_sex_metadata INTEGER,
    studies_sex_inferrable INTEGER,
    studies_with_sex_analysis INTEGER,

    avg_metadata_completeness FLOAT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_studies_organism ON studies(organism);
CREATE INDEX IF NOT EXISTS idx_studies_has_sex ON studies(has_sex_metadata);
CREATE INDEX IF NOT EXISTS idx_studies_inferrable ON studies(sex_inferrable);
CREATE INDEX IF NOT EXISTS idx_studies_geo_accession ON studies(geo_accession);
CREATE INDEX IF NOT EXISTS idx_samples_study_id ON samples(study_id);
CREATE INDEX IF NOT EXISTS idx_disease_mappings_study_id ON disease_mappings(study_id);
CREATE INDEX IF NOT EXISTS idx_disease_mappings_category ON disease_mappings(disease_category);

-- Enable Row Level Security (optional but recommended for production)
-- ALTER TABLE studies ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE samples ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE disease_mappings ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE completeness_snapshots ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access (if RLS is enabled)
-- CREATE POLICY "Allow public read access on studies" ON studies FOR SELECT USING (true);
-- CREATE POLICY "Allow public read access on samples" ON samples FOR SELECT USING (true);
-- CREATE POLICY "Allow public read access on disease_mappings" ON disease_mappings FOR SELECT USING (true);
-- CREATE POLICY "Allow public read access on completeness_snapshots" ON completeness_snapshots FOR SELECT USING (true);
