fn main() {
    env_logger::init();
    let json_string = r#"{"a1": "a\\", "b1": "string with \\\"so called\\\\\" double quotes", "a":null,"b":123,"c":24562472.12346757,"d":"a string","e":[1,2,3],"f":["a","b","c"],"g":{"a":{"b":1},"c":[{"x":1},{"y":2}],"d":[[1,2],[3,4]]}}"#;

    match pollster::block_on(parser_core::run(json_string)) {
        Ok((oc_pairs, structural_indexes)) => {
            // for bitmap output:
            /*  let gpu: Vec<u32> = output
            .iter()
            .enumerate()
            .flat_map(|(word_idx, word)| {
                (0..32u32).filter_map(move |bit| {
                    if (word >> bit) & 1 == 1 {
                        Some(word_idx as u32 * 32 + bit)
                    } else {
                        None
                    }
                })
            })
            .collect(); */
            let gpu: Vec<u32> = oc_pairs.iter().copied().collect();
            println!("gpu {:?}", gpu);
            println!("structural_indexes {:?}", structural_indexes);
            let bytes = json_string.as_bytes();
            for &idx in &[203u32, 205, 207, 208, 209, 210] {
                println!("pos {}: {:?}", idx, bytes[idx as usize] as char);
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}
