use std::ops::Add;

pub fn cumulative_sum<T>(collection: &[T], zero: T) -> Vec<T>
where
    T: Add<Output = T> + Copy,
{
    std::iter::once(zero)
        .chain(collection.iter().scan(zero, |state, &x| {
            *state = *state + x;
            Some(*state)
        }))
        .collect()
}


#[cfg(test)]
mod tests {
    use crate::helpers::cumulative_sum;
    use insta::assert_debug_snapshot;

    #[test]
    fn test_cumulative_sum() {
        let plain = vec![1, 2, 3, 4, 5];
        let sum = cumulative_sum(&plain, 0);
        assert_debug_snapshot!(sum)
    }
}
