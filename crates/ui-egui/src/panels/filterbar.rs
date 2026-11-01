//! The library filter bar above the grid: rating (≥ / = / ≤), flags, colour labels, kind,
//! edited state and camera / lens / keyword pickers — every control drives `library.filter`.
//! "Save as Smart Album…" keeps the current view as a live album; "Clear" resets the filter.
//!
//! The controls keep their groups together and flow onto as many rows as the grid's width needs,
//! so opening a side panel (which narrows the grid) never hides them. Row packing uses each
//! group's width measured on the previous frame, so it adapts to the current labels.

use egui::{Color32, Rect, Sense, Stroke, StrokeKind, vec2};
use lightcraft_catalog::{ColorLabel, Flag, MediaKind, RatingOp};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::icons::{Icon, paint};
use crate::theme::Tokens;
use crate::widgets::register;

/// One row's height (points); the bar grows with the number of wrapped rows.
pub const HEIGHT: f32 = 40.0;
/// Height of the controls inside a row.
const ROW_H: f32 = HEIGHT - 10.0;
/// Horizontal/vertical padding inside the bar.
const PAD_X: f32 = 16.0;
const PAD_Y: f32 = 5.0;
/// Space between groups on a row.
const GROUP_GAP: f32 = 14.0;
/// Number of groups (see `GROUPS` ids in [`show`]).
const GROUPS: usize = 9;

/// Colour-label swatches (UI colours, our own choice).
pub use crate::theme::label_color;

fn filter(app: &mut LightcraftApp, patch: Value) {
    let _ = app.run("library.filter", patch);
}

/// A small label before a group of controls.
fn caption(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new(crate::i18n::tr(text)).font(t.font(12.0)).color(t.text_dim));
}

/// A square toggle with an icon or a colour swatch.
fn toggle(ui: &mut egui::Ui, id: &str, on: bool, tip: &str, draw: impl FnOnce(&egui::Painter, Rect, Color32)) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(26.0, 26.0), Sense::click());
    register(ui.ctx(), format!("filter:{id}"), r);
    let p = ui.painter();
    if on {
        p.rect_filled(r, 5.0, t.tool_active);
        p.rect_stroke(r, 5.0, Stroke::new(1.0, t.accent), StrokeKind::Inside);
    } else if resp.hovered() {
        p.rect_filled(r, 5.0, t.hover);
    }
    draw(
        p,
        r.shrink(5.0),
        if on {
            t.text
        } else if resp.hovered() {
            t.text_label
        } else {
            t.icon
        },
    );
    resp.on_hover_text(crate::i18n::tr(tip))
}

/// A dropdown showing `current`; `items` are (label, filter patch, selected).
fn picker(app: &mut LightcraftApp, ui: &mut egui::Ui, id: &str, current: &str, active: bool, items: Vec<(String, Value, bool)>) {
    let t = Tokens::get(ui.ctx());
    let r = crate::widgets::dropdown(
        ui,
        &format!("filter-{id}"),
        if active && matches!(id, "camera" | "lens" | "keyword") { current } else { crate::i18n::tr(current) },
        t.font(12.5),
        if active { t.text } else { t.text_label },
    );
    if active {
        ui.painter().rect_stroke(r.rect.expand(2.0), 4.0, Stroke::new(1.0, t.accent.gamma_multiply(0.7)), StrokeKind::Outside);
    }
    let mut chosen = None;
    egui::Popup::menu(&r).show(|ui| {
        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
            for (label, patch, sel) in &items {
                if ui
                    .selectable_label(
                        *sel,
                        if matches!(id, "camera" | "lens" | "keyword") && patch.get(id).is_some_and(|value| !value.is_null()) {
                            label.as_str()
                        } else {
                            crate::i18n::tr(label)
                        },
                    )
                    .clicked()
                {
                    chosen = Some(patch.clone());
                }
            }
        });
    });
    if let Some(p) = chosen {
        filter(app, p);
    }
}

/// A value shorter than 22 characters (with an ellipsis when cut).
fn short(s: String) -> String {
    if s.chars().count() > 22 { format!("{}…", s.chars().take(21).collect::<String>()) } else { s }
}

/// Initial width guesses so the first frame already wraps sensibly.
fn default_widths() -> Vec<f32> {
    vec![200.0, 118.0, 190.0, 100.0, 120.0, 100.0, 90.0, 100.0, 320.0]
}

pub fn show(app: &mut LightcraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let f = app.session.filter.clone();
    let values = app.caches.filter_values(&app.session.catalog);
    let (cameras, lenses, keywords) = (values.cameras.clone(), values.lenses.clone(), values.keywords.clone());
    let widths_key = egui::Id::new("filterbar-group-widths");
    let mut widths: Vec<f32> = ui.ctx().data(|d| d.get_temp(widths_key)).unwrap_or_else(default_widths);
    if widths.len() != GROUPS {
        widths = default_widths();
    }
    let avail = (ui.available_width() - 2.0 * PAD_X).max(0.0);
    // greedy row packing: put a group on the current row if it fits, else start a new one
    let mut rows: Vec<Vec<usize>> = vec![Vec::new()];
    let mut row_w = 0.0_f32;
    for gi in 0..GROUPS {
        let w = widths.get(gi).copied().unwrap_or(0.0);
        let first = rows.last().is_none_or(Vec::is_empty);
        let need = if first { w } else { GROUP_GAP + w };
        if !first && row_w + need > avail {
            rows.push(Vec::new());
            row_w = w;
        } else {
            row_w += need;
        }
        if let Some(row) = rows.last_mut() {
            row.push(gi);
        }
    }
    let nrows = rows.len().max(1);
    let bar_h = nrows as f32 * ROW_H + 2.0 * PAD_Y;
    let (bar, _) = ui.allocate_exact_size(vec2(ui.available_width(), bar_h), Sense::hover());
    ui.painter().rect_filled(bar, 0.0, t.chrome);
    ui.painter().line_segment([bar.left_bottom(), bar.right_bottom()], Stroke::new(1.0, t.divider));
    register(ui.ctx(), "filterbar", bar);

    // kind / edited labels (shared by their groups)
    let kind_label = match f.kind {
        None => "All types",
        Some(MediaKind::Image) => "Photos",
        Some(MediaKind::Raw) => "Raw",
        Some(MediaKind::Video) => "Videos",
    };
    let kind_label = match f.merged.as_deref() {
        Some("hdr") => "HDR",
        Some("panorama") => "Panoramas",
        Some("hdrPanorama") => "HDR Panoramas",
        Some(_) => "Merged",
        None => kind_label,
    };
    let edited_label = match f.edited {
        None => "Edited or not",
        Some(true) => "Edited",
        Some(false) => "Unedited",
    };
    let mut chosen_labels: Vec<ColorLabel> = f.labels.clone();
    if let Some(l) = f.label
        && !chosen_labels.contains(&l)
    {
        chosen_labels.push(l);
    }

    for (ri, items) in rows.iter().enumerate() {
        let row_rect = Rect::from_min_size(egui::pos2(bar.left() + PAD_X, bar.top() + PAD_Y + ri as f32 * ROW_H), vec2(avail, ROW_H));
        let mut row_ui = ui.new_child(egui::UiBuilder::new().max_rect(row_rect).layout(egui::Layout::left_to_right(egui::Align::Center)));
        row_ui.spacing_mut().item_spacing.x = GROUP_GAP;
        for &gi in items {
            let measured = row_ui
                .horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    match gi {
                        // rating: operator + stars
                        0 => {
                            caption(ui, "Rating");
                            let op_label = match f.rating_op {
                                RatingOp::AtLeast => "≥",
                                RatingOp::Exactly => "=",
                                RatingOp::AtMost => "≤",
                            };
                            picker(
                                app,
                                ui,
                                "ratingOp",
                                op_label,
                                false,
                                [
                                    ("≥  at least", "atLeast", RatingOp::AtLeast),
                                    ("=  exactly", "exactly", RatingOp::Exactly),
                                    ("≤  at most", "atMost", RatingOp::AtMost),
                                ]
                                .into_iter()
                                .map(|(l, k, op)| (l.to_string(), json!({"ratingOp": k}), f.rating_op == op))
                                .collect(),
                            );
                            for i in 1..=5u8 {
                                let (r, resp) = ui.allocate_exact_size(vec2(17.0, 26.0), Sense::click());
                                register(ui.ctx(), format!("filter:star{i}"), r);
                                let filled = i <= f.rating;
                                let c = if filled {
                                    t.star
                                } else if resp.hovered() {
                                    t.text_label
                                } else {
                                    t.icon
                                };
                                paint(
                                    ui.painter(),
                                    Rect::from_center_size(r.center(), vec2(14.0, 14.0)),
                                    if filled { Icon::StarFilled } else { Icon::Star },
                                    c,
                                );
                                if resp.clicked() {
                                    filter(app, json!({"rating": if f.rating == i { 0 } else { i }}));
                                }
                            }
                        }
                        // flags (one at a time; click again to clear)
                        1 => {
                            caption(ui, "Flag");
                            for (id, icon, flag, tip) in [
                                ("pick", Icon::FlagPick, Flag::Pick, "Picked"),
                                ("reject", Icon::FlagReject, Flag::Reject, "Rejected"),
                                ("none", Icon::Circle, Flag::None, "Unflagged"),
                            ] {
                                let on = f.flag == Some(flag);
                                let color = match flag {
                                    Flag::Pick if on => t.pick,
                                    Flag::Reject if on => t.reject,
                                    _ => Color32::TRANSPARENT,
                                };
                                let resp =
                                    toggle(ui, id, on, tip, |p, r, c| paint(p, r, icon, if color == Color32::TRANSPARENT { c } else { color }));
                                if resp.clicked() {
                                    filter(app, json!({"flag": if on { Value::Null } else { json!(id) }}));
                                }
                            }
                        }
                        // colour labels: several can be on at once (any of them matches)
                        2 => {
                            caption(ui, "Label");
                            for l in ColorLabel::ALL {
                                let on = chosen_labels.contains(&l);
                                let name = format!("{l:?}").to_lowercase();
                                let tip = crate::i18n::tr_format!(
                                    "{} label (click more labels to show any of them)",
                                    crate::i18n::color_label(&app.session.catalog, l)
                                );
                                let resp = toggle(ui, &format!("label-{name}"), on, &tip, |p, r, _| {
                                    p.circle_filled(r.center(), 5.5, label_color(l));
                                });
                                if resp.clicked() {
                                    let mut next = chosen_labels.clone();
                                    if on {
                                        next.retain(|x| *x != l);
                                    } else {
                                        next.push(l);
                                    }
                                    next.sort();
                                    let names: Vec<String> = next.iter().map(|x| format!("{x:?}").to_lowercase()).collect();
                                    filter(app, json!({"label": Value::Null, "labels": names}));
                                }
                            }
                        }
                        3 => picker(
                            app,
                            ui,
                            "kind",
                            kind_label,
                            f.kind.is_some() || f.merged.is_some(),
                            [
                                ("All types", Value::Null, None),
                                ("Photos", json!("image"), Some(MediaKind::Image)),
                                ("Raw", json!("raw"), Some(MediaKind::Raw)),
                                ("Videos", json!("video"), Some(MediaKind::Video)),
                            ]
                            .into_iter()
                            .map(|(l, v, k)| (l.to_string(), json!({"kind": v, "merged": null}), f.kind == k && f.merged.is_none()))
                            .chain(
                                [("HDR", "hdr"), ("Panoramas", "panorama"), ("HDR Panoramas", "hdrPanorama")]
                                    .into_iter()
                                    .map(|(l, m)| (l.to_string(), json!({"kind": null, "merged": m}), f.merged.as_deref() == Some(m))),
                            )
                            .collect(),
                        ),
                        4 => picker(
                            app,
                            ui,
                            "edited",
                            edited_label,
                            f.edited.is_some(),
                            [("Edited or not", Value::Null, None), ("Edited", json!(true), Some(true)), ("Unedited", json!(false), Some(false))]
                                .into_iter()
                                .map(|(l, v, e)| (l.to_string(), json!({"edited": v}), f.edited == e))
                                .collect(),
                        ),
                        // metadata pickers
                        5 => {
                            let current = f.camera.clone();
                            let mut items = vec![("Any camera".to_string(), json!({"camera": Value::Null}), current.is_none())];
                            items.extend(cameras.iter().map(|v| (v.clone(), json!({"camera": v}), current.as_deref() == Some(v.as_str()))));
                            picker(app, ui, "camera", &short(current.unwrap_or_else(|| "Camera".to_string())), f.camera.is_some(), items);
                        }
                        6 => {
                            let current = f.lens.clone();
                            let mut items = vec![("Any lens".to_string(), json!({"lens": Value::Null}), current.is_none())];
                            items.extend(lenses.iter().map(|v| (v.clone(), json!({"lens": v}), current.as_deref() == Some(v.as_str()))));
                            picker(app, ui, "lens", &short(current.unwrap_or_else(|| "Lens".to_string())), f.lens.is_some(), items);
                        }
                        7 => {
                            let current = f.keyword.clone();
                            let mut items = vec![("Any keyword".to_string(), json!({"keyword": Value::Null}), current.is_none())];
                            items.extend(keywords.iter().map(|v| (v.clone(), json!({"keyword": v}), current.as_deref() == Some(v.as_str()))));
                            picker(app, ui, "keyword", &short(current.unwrap_or_else(|| "Keyword".to_string())), f.keyword.is_some(), items);
                        }
                        // save / clear / presets
                        _ => {
                            let filtering = app.session.filter != Default::default();
                            let save = ui.add_enabled(
                                filtering,
                                egui::Button::new(egui::RichText::new(crate::i18n::tr("Save as Smart Album…")).font(t.font(12.5))),
                            );
                            register(ui.ctx(), "button:filterSave", save.rect);
                            if save.clicked() {
                                let _ = app.run("dialog.newSmartAlbum", json!({}));
                            }
                            let clear =
                                ui.add_enabled(filtering, egui::Button::new(egui::RichText::new(crate::i18n::tr("Clear")).font(t.font(12.5))));
                            register(ui.ctx(), "button:filterClear", clear.rect);
                            if clear.clicked() {
                                app.ui.search.clear();
                                let _ = app.run("library.clearFilter", json!({}));
                            }
                            let presets = ui.button(egui::RichText::new(crate::i18n::tr("Presets ▾")).font(t.font(12.5)));
                            register(ui.ctx(), "button:filterPresets", presets.rect);
                            egui::Popup::menu(&presets).show(|ui| {
                                let names: Vec<String> = app.session.filter_presets.iter().map(|f| f.name.clone()).collect();
                                for n in &names {
                                    let r = ui.button(n);
                                    if r.clicked() {
                                        let _ = app.run("filter.applyPreset", json!({"name": n}));
                                    }
                                    r.context_menu(|ui| {
                                        if ui.button(crate::i18n::tr("Delete Preset")).clicked() {
                                            let _ = app.run("filter.deletePreset", json!({"name": n}));
                                        }
                                    });
                                }
                                if names.is_empty() {
                                    ui.label(egui::RichText::new(crate::i18n::tr("No filter presets yet")).color(t.text_dim));
                                }
                                ui.separator();
                                if ui.add_enabled(filtering, egui::Button::new(crate::i18n::tr("Save Current Filter as Preset…"))).clicked() {
                                    crate::panels::dialogs::prompt(
                                        app,
                                        "Save Filter Preset",
                                        "Preset name",
                                        "",
                                        "filter.savePreset",
                                        json!({}),
                                        "name",
                                    );
                                }
                            });
                        }
                    }
                })
                .response
                .rect
                .width();
            if let Some(w) = widths.get_mut(gi) {
                *w = measured;
            }
        }
    }
    ui.ctx().data_mut(|d| d.insert_temp(widths_key, widths));
}
