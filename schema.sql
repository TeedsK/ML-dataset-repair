DROP TABLE IF EXISTS repairs CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS domains CASCADE;
DROP TABLE IF EXISTS violations CASCADE;
DROP TABLE IF EXISTS noisy_cells CASCADE;
DROP TABLE IF EXISTS cells CASCADE;
DROP TABLE IF EXISTS ext_dict CASCADE;


CREATE TABLE cells (
    tid BIGINT,
    attr TEXT,
    val TEXT,
    is_noisy BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (tid, attr)
);
COMMENT ON TABLE cells IS 'stores each cell from the input dataset';
COMMENT ON COLUMN cells.tid IS 'unique id for the original tuple/row';
COMMENT ON COLUMN cells.attr IS 'name of the attribute/column';
COMMENT ON COLUMN cells.val IS 'original value of the cell';
COMMENT ON COLUMN cells.is_noisy IS 'flag set by error detectors if the cell is potentially incorrect';

CREATE TABLE domains (
    tid BIGINT,
    attr TEXT,
    candidate_val TEXT,
    PRIMARY KEY (tid, attr, candidate_val),
    FOREIGN KEY (tid, attr) REFERENCES cells(tid, attr) ON DELETE CASCADE
);

COMMENT ON TABLE domains IS 'stores candidate/domain repair values for each cell which is  populated after domain pruning';
COMMENT ON COLUMN domains.candidate_val IS 'potential correct value for the cell id';

CREATE TABLE features (
    tid BIGINT,
    attr TEXT,
    candidate_val TEXT,
    feature TEXT,
    PRIMARY KEY (tid, attr, candidate_val, feature),
    FOREIGN KEY (tid, attr, candidate_val) REFERENCES domains(tid, attr, candidate_val) ON DELETE CASCADE
);
COMMENT ON TABLE features IS 'stores features used in the factor graph which provides evidence for or against candidate values';
COMMENT ON COLUMN features.candidate_val IS 'specific candidate value this feature relates to';
COMMENT ON COLUMN features.feature IS 'id for the feature';

CREATE TABLE violations (
    violation_id SERIAL PRIMARY KEY,
    constraint_id INT NOT NULL,
    tids BIGINT[] NOT NULL
);

COMMENT ON TABLE violations IS 'logs detected integrity constraint violations';
COMMENT ON COLUMN violations.constraint_id IS 'id for the denial constraint being violated';
COMMENT ON COLUMN violations.tids IS 'array of tuple id participating in this violation instance';

CREATE TABLE noisy_cells (
    tid BIGINT,
    attr TEXT,
    detection_method TEXT,
    PRIMARY KEY (tid, attr),
    FOREIGN KEY (tid, attr) REFERENCES cells(tid, attr) ON DELETE CASCADE
);

COMMENT ON TABLE noisy_cells IS 'stores cells flagged as potentially wrong by the detectors';


CREATE TABLE ext_dict (
    dict_id TEXT NOT NULL,
    ext_tid TEXT NOT NULL,
    ext_attr TEXT NOT NULL,
    ext_val TEXT,
    PRIMARY KEY (dict_id, ext_tid, ext_attr)
);

COMMENT ON TABLE ext_dict IS 'table to store external dictionary/knowledge base data for matching.';

CREATE TABLE repairs (
    tid BIGINT,
    attr TEXT,
    repaired_val TEXT,
    confidence REAL,
    PRIMARY KEY (tid, attr),
    FOREIGN KEY (tid, attr) REFERENCES cells(tid, attr) ON DELETE CASCADE
);

COMMENT ON TABLE repairs IS 'stores the final repairs after inference.';
COMMENT ON COLUMN repairs.repaired_val IS 'the cleaned value proposed';
COMMENT ON COLUMN repairs.confidence IS 'the computed marginal probability';


CREATE INDEX idx_cells_tid ON cells(tid);
CREATE INDEX idx_cells_attr ON cells(attr);
CREATE INDEX idx_cells_val ON cells(val);
CREATE INDEX idx_cells_is_noisy ON cells(is_noisy);

CREATE INDEX idx_domains_tid_attr ON domains(tid, attr);

CREATE INDEX idx_features_tid_attr_candidate ON features(tid, attr, candidate_val);
CREATE INDEX idx_features_feature ON features(feature);

CREATE INDEX idx_violations_constraint ON violations(constraint_id);
CREATE INDEX idx_violations_tids ON violations USING GIN(tids);

CREATE INDEX idx_ext_dict_val ON ext_dict(ext_val);

COMMIT;