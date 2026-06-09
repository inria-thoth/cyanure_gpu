Release 1.0.3

- Relax dependency constraints: remove the numpy and torch upper bounds.
- Require scikit-learn >= 1.6 and migrate to its API (validate_data, dataclass tags, ensure_all_finite).
- Support numpy 2.x (use np.exceptions.VisibleDeprecationWarning).
- Fix several data-processing bugs: reject sparse input before conversion in preprocess, correct the dtype check in is_multilabel, correct the finiteness check in check_is_finite, and report the actual feature count in the inference shape error.
