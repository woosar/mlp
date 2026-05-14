use std::path::PathBuf;

pub struct FileSystemArchivist;

pub struct FileSystemMetadata {
    id: String,
    filename: String,
    collection: String,
    root: PathBuf,
}
