//! Library commands: view source, filter/sort, selection, ratings/flags/labels, rotate, delete,
//! metadata, albums, import.

use lightcraft_catalog::{Album, AlbumId, ColorLabel, CopyrightStatus, Filter, Flag, GroupBy, Op, PhotoId, Sort, SortKey};
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, bool_or, cmd, has_active, has_library_selection, has_selection, ok, str_param};
use crate::{LibrarySource, Result, Selection, Session};

mod filters;
mod gps;
mod import;
mod selection;
mod specs;

pub(crate) use filters::covers_other_disks;
pub use import::{ImportRequest, import_batch_done, import_params, list_auto_import_folder};
pub use specs::specs;

#[cfg(test)]
#[path = "library/tests_relative_ratings.rs"]
mod relative_rating_tests;

#[cfg(test)]
#[path = "library/tests_flag_cycle.rs"]
mod flag_cycle_tests;

#[cfg(test)]
#[path = "library/tests_rated_out.rs"]
mod rated_out_tests;

#[cfg(test)]
#[path = "library/tests_clear_label.rs"]
mod clear_label_tests;

#[cfg(test)]
#[path = "library/tests_enable_filters.rs"]
mod enable_filters_tests;

#[cfg(test)]
#[path = "library/tests_source_album_filter.rs"]
mod source_album_filter_tests;

/// The photos an import batch put in (or found in) the library for an album: the imported and
/// restored ones, plus the duplicates the library already had (not those in Recently Deleted).
pub fn import_album_members(s: &Session, report: &crate::import::ImportReport) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    let existing = report
        .duplicates
        .iter()
        .filter(|d| !d.existing_deleted)
        .filter_map(|d| d.existing)
        .filter(|id| s.catalog.photo(PhotoId(*id)).is_some_and(|p| !p.deleted));
    for id in report.imported.iter().chain(&report.restored).copied().chain(existing) {
        if !out.contains(&id) {
            out.push(id);
        }
    }
    out
}
