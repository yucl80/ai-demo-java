package com.yucl.demo.spring.ai.graph.nebula;

public class NL2Cypher {

    public static String PROMP = "Task:Generate NebulaGraph Cypher statement to query a graph database.\n\n"+
    "Instructions:\n\n"+
    "First, generate cypher then convert it to NebulaGraph Cypher dialect(rather than standard):\n"+
    "1. it requires explicit label specification only when referring to node properties: v.`Foo`.name\n"+
    "2. note explicit label specification is not needed for edge properties, so it's e.name instead of e.`Bar`.name\n"+
    "3. it uses double equals sign for comparison: `==` rather than `=`\nFor instance:\n```diff\n< MATCH (p:person)-[e:directed]->(m:movie) WHERE m.name = 'The Godfather II'\n< RETURN p.name, e.year, m.name;\n---\n> MATCH (p:`person`)-[e:directed]->(m:`movie`) WHERE m.`movie`.`name` == 'The Godfather II'\n> RETURN p.`person`.`name`, e.year, m.`movie`.`name`;\n```\n\nUse only the provided relationship types and properties in the schema.\nDo not use any other relationship types or properties that are not provided.\n" + 
    "Schema:\nNode properties: [{'tag': 'movie', 'properties': [('name', 'string')]}, {'tag': 'person', 'properties': [('name', 'string'), ('birthdate', 'string')]}]\n" + 
    "Edge properties: [{'edge': 'acted_in', 'properties': [('acted_time', 'date')]}, {'edge': 'impl', 'properties': []}, {'edge': 'uses', 'properties': []}]\n" + 
    "Relationships: ['(:person)-[:acted_in]->(:movie)']\n\n" + 
    "Note: Do not include any explanations or apologies in your responses.\n" + 
    "Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.\n" + 
    "Do not include any text except the generated Cypher statement.\n\nThe question is:\n {question}";

    public static String DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL = "Generate NebulaGraph query from natural language.\n" + 
    "Use only the provided relationship types and properties in the schema.\n" + 
    "Do not use any other relationship types or properties that are not provided.\n" + 
    "Schema:\n" + 
    "---\n" + 
    "{schema}\n" + 
    "---\n" + 
    "Note: NebulaGraph speaks a dialect of Cypher, comparing to standard Cypher:\n" + 
    "\n" + 
    "1. it uses double equals sign for comparison: `==` rather than `=`\n" + 
    "2. it needs explicit label specification when referring to node properties, i.e.\n" + 
    "v is a variable of a node, and we know its label is Foo, v.`foo`.name is correct\n" + 
    "while v.name is not.\n" + 
    "\n" + 
    "For example, see this diff between standard and NebulaGraph Cypher dialect:\n" + 
    "```diff\n" + 
    "< MATCH (p:person)-[:directed]->(m:movie) WHERE m.name = 'The Godfather'\n" + 
    "< RETURN p.name;\n" + 
    "---\n" + 
    "> MATCH (p:`person`)-[:directed]->(m:`movie`) WHERE m.`movie`.`name` == 'The Godfather'\n" + 
    "> RETURN p.`person`.`name`;\n" + 
    "```\n" + 
    "\n" + 
    "Question: {query_str}\n" + 
    "\n" + 
    "NebulaGraph Cypher dialect query:";


}
