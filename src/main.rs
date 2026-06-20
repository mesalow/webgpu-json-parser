use std::{
    fs::{File, read, read_to_string},
    iter::Enumerate,
    str::Chars,
};

use crate::QueryKey::{ArrayIndex, ObjectKey};
use std::string::String;

fn main() {
    env_logger::init();
    // let json_string = r#"{"a1": "a\\", "b1": "string with \\\"so called\\\\\" double quotes", "a":null,"b":123,"c":24562472.12346757,"d":"a string","e":[1,2,3],"f":["a","b","c"],"g":{"a":{"b":1},"c":[{"x":1},{"y":2}],"d":[[1,2],[3,4]]}}"#;

    //let json_string = r#"[{"a1": "a\\", "b1": "string with \\\"so called\\\\\" double quotes", "a3":null,"b":123,"c":24562472.12346757,"d":"a string","e":[1,2,3],"f":["a4","b","c"],"g":{"a":{"x":1,"y":"hi"},"c":[{"x":1},{"y":2}, {"z": [11,22,33]}],"d":[[1,2],[3,4]]}}]"#;

    let file_content = read_to_string("./twitter_sample_large_record.json");
    let json_string = file_content.unwrap();

    let query = "$[0].source";
    println!("reading in {:?}", json_string);
    println!("querying {:?}", query);
    match pollster::block_on(parser_core::run(json_string.as_str())) {
        Ok((oc_pairs, structural_indexes)) => {
            // for bitmap output:
            /* let gpu: Vec<u32> = output
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
            println!("oc_pairs_vec {:?}", oc_pairs);
            println!("structural_indexes {:?}", structural_indexes);

            let value = run_query(json_string.as_str(), oc_pairs, structural_indexes, query);
            println!("read value {:?}", value);
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}

#[derive(Debug)]
enum JsonPrimitive {
    String(String),
    Null,
    Bool(bool),
    Number(f64),
}

struct Parser {
    json_string: String,
    structural_indexes: Vec<u32>,
    open_close_pairs: Vec<u32>,
    current_structural_index: usize,
}

impl Parser {
    fn get_current_structural(&self) -> &str {
        let current_structural_index =
            self.structural_indexes[self.current_structural_index] as usize;
        &self.json_string[current_structural_index..current_structural_index + 1]
    }

    fn peek_next_structural(&self) -> &str {
        let structural = self.structural_indexes[self.current_structural_index + 1] as usize;
        &self.json_string[structural..structural + 1]
    }

    fn next_structural(&mut self) -> &str {
        self.current_structural_index = &self.current_structural_index + 1;
        let structural = self.structural_indexes[self.current_structural_index] as usize;
        &self.json_string[structural..structural + 1]
    }

    fn get_object_key_slice(&self) -> &str {
        let start_struct_idx = self.structural_indexes[self.current_structural_index] as usize; // the { or ,
        let end_struct_idx = self.structural_indexes[self.current_structural_index + 1] as usize; // the :
        let current_slice = &self.json_string[(start_struct_idx + 1)..end_struct_idx]; // the key in between with "", not escaped!
        println!(
            "GETKEY: json idxs {:?} -> {:?}",
            start_struct_idx + 1,
            end_struct_idx
        );

        let trimmed = current_slice.trim();
        println!(
            "GETKEY: current key {:?} for index {:?}",
            trimmed, self.current_structural_index
        );
        trimmed
    }

    fn get_value(&self, current_index_in_structural: usize) -> &str {
        let start_struct_idx = self.structural_indexes[current_index_in_structural] as usize; // start of object or array or ','
        let end_struct_idx = self.structural_indexes[current_index_in_structural + 1] as usize; // ','
        let current_slice = &self.json_string[(start_struct_idx + 1)..end_struct_idx]; // the value in between with "", not escaped!

        current_slice
    }

    fn assert_current_is_colon(&self) {
        println!("current structural {:?}", self.get_current_structural());
        if self.get_current_structural() != ":" {
            panic!(
                "ERROR: expected structural ':', got {:?}",
                self.get_current_structural()
            );
        }
    }

    fn assert_current_is_bracket(&self) {
        println!("current structural {:?}", self.get_current_structural());
        if self.get_current_structural() != "[" {
            panic!(
                "ERROR: expected structural '[', got {:?}",
                self.get_current_structural()
            );
        }
    }

    fn assert_current_is_brace(&self) {
        println!("current structural {:?}", self.get_current_structural());
        if self.get_current_structural() != "{" {
            panic!(
                "ERROR: expected structural '{{', got {:?}",
                self.get_current_structural()
            );
        }
    }

    fn jump_object_or_array(&mut self) {
        let idx_of_close = self.open_close_pairs[(self.current_structural_index as usize) + 1];
        let next_struct_idx = (idx_of_close + 1) as usize; // we don't want to look at the close char but the next structural as the basis, probably ','

        println!(
            "jump object or array: setting current index from {:?} to {:?}",
            self.current_structural_index, next_struct_idx
        );
        self.current_structural_index = next_struct_idx;
    }

    fn parse(&mut self, query: Vec<QueryKey>) -> JsonPrimitive {
        // we are at an index in structural array: some structural char in the original string
        // we are either inside an object or inside an array
        // if inside object, we need to find the key

        for query_key in query {
            match query_key {
                ArrayIndex(index) => {
                    self.assert_current_is_bracket();

                    // query $f.1 -> find key f which is an array, then same as in object: go through commas that are structural
                    // if next struct is comma and index is not reached, consume
                    // if next struct is not comma, it's an object / array => jump to next
                    println!("============");
                    println!("ARRAY: get index {:?}", index);
                    let mut looked_at_index = 0;

                    loop {
                        if looked_at_index == index {
                            break; // don't need to do anything if we have arrived at the index
                        }
                        println!("looking at index {:?}", looked_at_index);
                        // current struct is [ or ,
                        let next_structural_peeked = self.peek_next_structural();
                        match next_structural_peeked {
                            "," => {
                                // primitive value, we can just continue
                                self.next_structural();
                                looked_at_index += 1;
                                continue;
                            }
                            "{" | "[" => {
                                // object or array, we need to jump
                                self.jump_object_or_array();
                                looked_at_index += 1;
                            }
                            _ => panic!("unexpected structural in array iteration"),
                        }
                    }
                    println!("ARRAY: while ended"); // if index == 0 we don't need to do anything
                    // we found it -> either we are done or we need to consume the next structural char to recurse into array or object

                    self.next_structural();
                    // if we are done, outside the loop we will subtract 1 to consume the value
                }
                ObjectKey(key) => {
                    self.assert_current_is_brace();

                    println!("============");
                    println!("QUERY: get KEY {:?}", key);
                    println!(
                        "QUERY: current structural {:?}",
                        self.get_current_structural()
                    );

                    while let current_key = self.get_object_key_slice()
                        && current_key != format!("\"{}\"", key)
                    {
                        println!(
                            "OBJECTKEY: current key {:?} did not match query key {:?}",
                            current_key,
                            format!("\"{}\"", key)
                        );
                        self.next_structural(); // move to colon
                        self.assert_current_is_colon();

                        let end_struct_idx =
                            self.structural_indexes[(self.current_structural_index)] as usize; // the ':'
                        let next_in_structural_idx =
                            self.structural_indexes[(self.current_structural_index) + 1] as usize;
                        let diff = next_in_structural_idx - end_struct_idx;
                        if diff == 1 {
                            // next structural immediately follows colon -> object or array, jump ahead via pairs
                            // next_in_structural_idx thus points to a OPEN bracket or brace, so we need to use that idx in the pairs (current_idx+2)
                            println!("OBJECTKEY: diff is 1, jumping object or array");

                            self.jump_object_or_array();
                        } else {
                            println!("OBJECTKEY: diff is not 1, jumping primitive");
                            println!(
                                "OBJECTKEY: setting current index from {:?} to {:?}",
                                self.current_structural_index,
                                self.current_structural_index + 1
                            );
                            // next struct is farther away, next value is a primitive -> go to next , OR it's the end of the object, let's not think about that yet
                            self.current_structural_index = self.current_structural_index + 1;
                        }
                    }
                    println!("OBJECTKEY: while ended");
                    // current key matched
                    println!("found key{:?}", self.get_object_key_slice());

                    // consume following ":"
                    self.next_structural();
                    self.assert_current_is_colon();

                    // now we read the value, two cases: either query is finished and this is the searched value
                    // or we need to recurse into the object / array

                    // for the recursion case, we need to consume the starting structural
                    // for the end case, we will just subtract 1 outside the loop
                    self.next_structural();
                }
            }
        }
        println!("PARSE: for ended {:?}", self.current_structural_index - 1);
        let value_slice = self.get_value((self.current_structural_index) - 1);
        println!("found value {:?}", value_slice);
        match value_slice.trim() {
            "true" => JsonPrimitive::Bool(true),
            "false" => JsonPrimitive::Bool(false),
            "null" => JsonPrimitive::Null,
            s => match s.parse::<f64>() {
                Ok(number) => JsonPrimitive::Number(number),
                Err(_) => JsonPrimitive::String(s.trim_matches('"').to_string()),
            },
        }
    }
}

fn run_query<'a>(
    json_string: &'a str,
    oc_pairs: Vec<u32>,
    structural_indexes: Vec<u32>,
    query: &'a str,
) -> JsonPrimitive {
    let parsed_query = parse_query(query);
    let mut parser = Parser {
        json_string: json_string.to_string(),
        open_close_pairs: oc_pairs,
        structural_indexes: structural_indexes,
        current_structural_index: 0,
    };
    let value = parser.parse(parsed_query);
    value
}

#[derive(Debug)]
enum QueryKey {
    ArrayIndex(u32),
    ObjectKey(String),
}

// either we have an array index "[i]" or an object key ".k"
// $ is root, so $[1] means the root is expected to be an array and we want the second element
// $.a.b[0] means we want the object at key a which has a field b which is an array and we want the first element
fn parse_query(query: &str) -> Vec<QueryKey> {
    // query is in ascii so we should be able to use query.chars()
    let mut parsed_query = vec![];
    let mut iterator = query.chars().peekable();
    while let Some(current_char) = iterator.next() {
        println!("current char {:?}", current_char);
        if current_char == '[' {
            let mut result = String::new();
            while let Some(next) = iterator.peek() {
                if *next == ']' {
                    break;
                }
                result.push(*next);
                iterator.next();
            }

            let index: u32 = result.trim().parse().unwrap();
            parsed_query.push(ArrayIndex(index));
        }
        if current_char == '.' {
            let mut result = "".to_string();

            while let Some(next) = iterator.peek()
                && *next != '.'
                && *next != '['
            {
                result.push_str(next.to_string().as_str());
                iterator.next();
            }
            let key = result.trim().to_string();
            parsed_query.push(ObjectKey(key));
        }
    }
    println!("parsed query {:?}", parsed_query);
    parsed_query
}
