use serde::{Deserialize, Serialize};

pub trait Archivist {
    type Metadata;
    type Error;
    fn write<S: Serialize + for<'a> Deserialize<'a>>(
        &self,
        data: &S,
        metadata: Self::Metadata,
    ) -> Result<(), Self::Error>;
    fn read<S: Serialize + for<'a> Deserialize<'a>>(
        &self,
        metadata: Self::Metadata,
    ) -> Result<S, Self::Error>;
}

#[derive(Serialize, Deserialize)]
pub struct Entity<T> {
    pub id: String,
    #[serde(flatten)]
    pub data: T,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestData {
        name: String,
        value: i32,
    }

    #[test]
    fn test_entity_serialization() {
        let entity = Entity {
            id: "123".to_string(),
            data: TestData {
                name: "test".to_string(),
                value: 42,
            },
        };

        let json = serde_json::to_string(&entity).unwrap();
        assert!(json.contains("\"id\":\"123\""));
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("\"value\":42"));

        let deserialized: Entity<TestData> = serde_json::from_str(&json).unwrap();
        assert_eq!(entity.id, deserialized.id);
        assert_eq!(entity.data, deserialized.data);
    }
}
