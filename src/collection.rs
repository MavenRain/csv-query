//! CSV collection: schema validation and row merging.
//!
//! Multiple CSV sources are loaded into a single [`CsvCollection`]
//! with a unified [`Schema`].  The first source establishes the
//! canonical header row; every subsequent source must have an
//! identical header or loading fails with `Error::SchemaMismatch`.
//! Rows from all sources are concatenated in source order.

use comp_cat_rs::effect::io::Io;
use csv_cat::reader::{self, ReaderConfig};
use csv_cat::row::Row;

use crate::error::Error;
use crate::source::ResolvedSource;

/// Column headers shared across all files in the collection.
#[derive(Debug, Clone)]
pub struct Schema(Vec<String>);

impl Schema {
    /// The column names.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.0
    }

}

/// A validated CSV collection with a unified schema and merged rows.
#[derive(Debug)]
pub struct CsvCollection {
    schema: Schema,
    rows: Vec<Row>,
}

impl CsvCollection {
    /// The shared schema.
    #[must_use]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// All rows from every source.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Number of rows in the collection.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// Load resolved sources into a schema-validated collection.
///
/// The first source establishes the schema (header row).  Every
/// subsequent source must have an identical header row or the
/// load fails with `Error::SchemaMismatch`.
///
/// # Errors
///
/// - `Error::NoFiles` if the source list is empty.
/// - `Error::SchemaMismatch` if any source's headers differ from
///   the first source's headers.
/// - `Error::Csv` or `Error::Io` if a file cannot be read or parsed.
pub fn load(sources: Vec<ResolvedSource>) -> Io<Error, CsvCollection> {
    Io::suspend(move || {
        let loaded: Vec<(Vec<String>, Vec<Row>)> = sources
            .into_iter()
            .map(load_one)
            .collect::<Result<_, _>>()?;

        loaded
            .split_first()
            .ok_or(Error::NoFiles)
            .and_then(|((first_headers, first_rows), rest)| {
                rest.iter().try_fold(
                    first_rows.clone(),
                    |acc, (headers, rows)| {
                        if headers == first_headers {
                            Ok(acc.into_iter().chain(rows.clone()).collect())
                        } else {
                            Err(Error::SchemaMismatch {
                                expected: first_headers.clone(),
                                found: headers.clone(),
                            })
                        }
                    },
                )
                .map(|all_rows| CsvCollection {
                    schema: Schema(first_headers.clone()),
                    rows: all_rows,
                })
            })
    })
}

/// Load a single resolved source, returning its headers and rows.
fn load_one(source: ResolvedSource) -> Result<(Vec<String>, Vec<Row>), Error> {
    match source {
        ResolvedSource::FilePath(path) => load_file(&path),
        ResolvedSource::RawData(data) => load_string(&data),
    }
}

/// Load a CSV file, extracting headers and data rows.
fn load_file(path: &str) -> Result<(Vec<String>, Vec<Row>), Error> {
    let headers = extract_headers_from_file(path)?;
    let rows = reader::read_all(path, ReaderConfig::new())
        .run()
        .map_err(Error::from)?;
    Ok((headers, rows))
}

/// Load CSV from a string, extracting headers and data rows.
fn load_string(data: &str) -> Result<(Vec<String>, Vec<Row>), Error> {
    let headers = extract_headers_from_string(data)?;
    let rows = reader::from_str(data, ReaderConfig::new())
        .run()
        .map_err(Error::from)?;
    Ok((headers, rows))
}

/// Read just the header row from a file.
fn extract_headers_from_file(path: &str) -> Result<Vec<String>, Error> {
    #[allow(unused_mut)]
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(csv_cat::error::CsvError::from)?;
    rdr.headers()
        .map(|h| h.iter().map(String::from).collect())
        .map_err(|e| Error::from(csv_cat::error::CsvError::from(e)))
}

/// Read just the header row from a string.
fn extract_headers_from_string(data: &str) -> Result<Vec<String>, Error> {
    #[allow(unused_mut)]
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(data.as_bytes());
    rdr.headers()
        .map(|h| h.iter().map(String::from).collect())
        .map_err(|e| Error::from(csv_cat::error::CsvError::from(e)))
}
