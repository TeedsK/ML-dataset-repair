-- File: schema.sql
-- Defines the core tables for the HoloClean implementation

-- Drop existing tables if they exist (for easy reset during development)
DROP TABLE IF EXISTS repairs CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS domains CASCADE;
DROP TABLE IF EXISTS violations CASCADE;
DROP TABLE IF EXISTS noisy_cells CASCADE;
DROP TABLE IF EXISTS cells CASCADE;
DROP TABLE IF EXISTS ext_dict CASCADE;


-- Main table storing individual cell values from the input dataset
CREATE TABLE cells (
    tid BIGINT,          -- Stable tuple identifier (e.g., original row index + 1)
    attr TEXT,           -- Attribute name (column header)
    val TEXT,            -- Cell value (as string)
    is_noisy BOOLEAN DEFAULT FALSE, -- Flag indicating if detected as potentially erroneous
    PRIMARY KEY (tid, attr) -- Each cell is unique per tuple and attribute
);
COMMENT ON TABLE cells IS 'Stores each cell (tid, attribute, value) from the input dataset.';
COMMENT ON COLUMN cells.tid IS 'Unique identifier for the original tuple/row.';
COMMENT ON COLUMN cells.attr IS 'Name of the attribute/column.';
COMMENT ON COLUMN cells.val IS 'Original value of the cell.';
COMMENT ON COLUMN cells.is_noisy IS 'Flag set by error detectors if the cell is potentially incorrect.';


-- Table storing the potential candidate values for each cell (after pruning)
CREATE TABLE domains (
    tid BIGINT,          -- Tuple identifier, links to cells
    attr TEXT,           -- Attribute name, links to cells
    candidate_val TEXT,  -- A potential correct value for the cell (tid, attr)
    PRIMARY KEY (tid, attr, candidate_val), -- Ensures each candidate is unique per cell
    FOREIGN KEY (tid, attr) REFERENCES cells(tid, attr) ON DELETE CASCADE -- Ensures candidates belong to existing cells
);
COMMENT ON TABLE domains IS 'Stores candidate repair values for each cell, populated after domain pruning.';
COMMENT ON COLUMN domains.candidate_val IS 'A potential correct value for the cell identified by (tid, attr).';


-- Table storing features associated with candidate values, used for probabilistic model
CREATE TABLE features (
    tid BIGINT,          -- Tuple identifier
    attr TEXT,           -- Attribute name
    candidate_val TEXT,  -- The candidate value this feature provides evidence for/against
    feature TEXT,        -- Description of the feature (e.g., co-occurrence, prior, external match, DC)
    PRIMARY KEY (tid, attr, candidate_val, feature), -- Feature definition is unique
    FOREIGN KEY (tid, attr, candidate_val) REFERENCES domains(tid, attr, candidate_val) ON DELETE CASCADE -- Feature must relate to a valid candidate
);
COMMENT ON TABLE features IS 'Stores features used in the factor graph, providing evidence for/against candidate values.';
COMMENT ON COLUMN features.candidate_val IS 'The specific candidate value this feature relates to.';
COMMENT ON COLUMN features.feature IS 'Identifier for the feature (e.g., "cooc_City=chicago", "prior_minimality").';


-- Table storing detected violations of integrity constraints (e.g., Denial Constraints)
CREATE TABLE violations (
    violation_id SERIAL PRIMARY KEY, -- Unique ID for this specific violation instance
    constraint_id INT NOT NULL,      -- Identifier linking to the specific constraint rule that was violated
    tids BIGINT[] NOT NULL           -- Array of tuple IDs involved in this violation instance
);
COMMENT ON TABLE violations IS 'Logs detected integrity constraint violations.';
COMMENT ON COLUMN violations.constraint_id IS 'Identifier for the denial constraint being violated.';
COMMENT ON COLUMN violations.tids IS 'Array of tuple IDs participating in this violation instance.';


-- Optional but recommended: Explicitly track cells identified as noisy by detectors
CREATE TABLE noisy_cells (
    tid BIGINT,
    attr TEXT,
    detection_method TEXT, -- Optional: Record which detector flagged this cell
    PRIMARY KEY (tid, attr), -- Cell is uniquely identified
    FOREIGN KEY (tid, attr) REFERENCES cells(tid, attr) ON DELETE CASCADE
);
COMMENT ON TABLE noisy_cells IS 'Explicitly stores cells flagged as potentially erroneous by various detectors.';


-- Table for storing external dictionary/knowledge base information (optional)
CREATE TABLE ext_dict (
    dict_id TEXT NOT NULL,        -- Identifier for the dictionary source (e.g., 'us_zip_codes')
    ext_tid TEXT NOT NULL,        -- Tuple identifier/key within the external dictionary
    ext_attr TEXT NOT NULL,       -- Attribute name within the external dictionary
    ext_val TEXT,                 -- Value in the external dictionary
    PRIMARY KEY (dict_id, ext_tid, ext_attr) -- Uniquely identify external data points
);
COMMENT ON TABLE ext_dict IS 'Optional table to store external dictionary/knowledge base data for matching.';


-- Table to store the final proposed repairs
CREATE TABLE repairs (
    tid BIGINT,
    attr TEXT,
    repaired_val TEXT, -- The value HoloClean proposes as the correction
    confidence REAL,   -- The marginal probability (confidence) of this repair
    PRIMARY KEY (tid, attr), -- Only one repair per cell
    FOREIGN KEY (tid, attr) REFERENCES cells(tid, attr) ON DELETE CASCADE
);
COMMENT ON TABLE repairs IS 'Stores the final repairs proposed by HoloClean after inference.';
COMMENT ON COLUMN repairs.repaired_val IS 'The cleaned value proposed by the system.';
COMMENT ON COLUMN repairs.confidence IS 'The computed marginal probability for the proposed repaired_val.';


-- === INDEXES ===
-- Indexing is crucial for performance, especially for joins and lookups

-- On cells table
CREATE INDEX idx_cells_tid ON cells(tid);
CREATE INDEX idx_cells_attr ON cells(attr);
CREATE INDEX idx_cells_val ON cells(val); -- Useful for finding co-occurrences
CREATE INDEX idx_cells_is_noisy ON cells(is_noisy);

-- On domains table
CREATE INDEX idx_domains_tid_attr ON domains(tid, attr); -- Covered by PK, but good practice

-- On features table
CREATE INDEX idx_features_tid_attr_candidate ON features(tid, attr, candidate_val); -- Covered by PK
CREATE INDEX idx_features_feature ON features(feature); -- If searching by feature type

-- On violations table
CREATE INDEX idx_violations_constraint ON violations(constraint_id);
CREATE INDEX idx_violations_tids ON violations USING GIN(tids); -- GIN index for array containment queries

-- On noisy_cells table
-- PK covers (tid, attr) lookup

-- On ext_dict table
CREATE INDEX idx_ext_dict_val ON ext_dict(ext_val); -- Useful for matching values

-- On repairs table
-- PK covers (tid, attr) lookup

COMMIT; -- Ensure changes are saved if run in a transaction block