# build_kg/config.py

SQL_QUERIES = {
    "SYNC_AUTHORS": "SELECT author_id as id, author_id as aid, name, h_index, cited_by_count as citations, last_updated FROM authors WHERE last_updated > ?",
    "SYNC_WORKS": "SELECT work_id as id, work_id as wid, title as name, title, year, citation_count as citations, concepts_text, keywords_text FROM works",
    "SYNC_INSTITUTIONS": "SELECT inst_id as id, inst_id as iid, name, cited_by_count as citations, last_updated FROM institutions WHERE last_updated > ?",
    "SYNC_SOURCES": "SELECT source_id as id, source_id as sid, display_name as name, type, cited_by_count as citations, last_updated FROM sources WHERE last_updated > ?",
    "SYNC_JOBS": "SELECT securityId as id, securityId as jid, job_name as name, skills, description, crawl_time FROM jobs WHERE crawl_time > ?",
    "GET_ALL_VOCAB": "SELECT id, id as vid, term as name, term, entity_type FROM vocabulary",
    "SYNC_AUTHORED_TOPOLOGY": """
        SELECT aship.id as sync_id, aship.author_id as aid, aship.work_id as wid, 
               aship.inst_id as iid, aship.source_id as sid, aship.pos_index, 
               aship.is_corresponding, aship.is_alphabetical, w.year
        FROM authorships aship
        JOIN works w ON aship.work_id = w.work_id
        WHERE aship.id > ?
    """
}

CYPHER_TEMPLATES = {
    "MERGE_AUTHOR": "UNWIND $data AS row MERGE (n:Author {id: row.id}) SET n.aid = row.aid, n.name = row.name, n.h_index = row.h_index, n.citations = row.citations",
    "MERGE_WORK": "UNWIND $data AS row MERGE (n:Work {id: row.id}) SET n.wid = row.wid, n.title = row.title, n.name = row.name, n.year = row.year, n.citations = row.citations",
    "MERGE_INSTITUTION": "UNWIND $data AS row MERGE (i:Institution {id: row.id}) SET i.iid = row.iid, i.name = row.name, i.citations = row.citations",
    "MERGE_SOURCE": "UNWIND $data AS row MERGE (s:Source {id: row.id}) SET s.sid = row.sid, s.name = row.name, s.type = row.type, s.citations = row.citations",
    "MERGE_JOB": "UNWIND $data AS row MERGE (j:Job {id: row.id}) SET j.jid = row.jid, j.name = row.name, j.skills = row.skills, j.description = row.description",
    "MERGE_VOCAB": "UNWIND $data AS row MERGE (v:Vocabulary {id: row.id}) SET v.vid = row.vid, v.term = row.term, v.type = row.entity_type",
    "LINK_AUTHORED_COMPLEX": """
        UNWIND $data AS row
        MATCH (a:Author {id: row.aid}), (w:Work {id: row.wid})
        MERGE (a)-[r:AUTHORED]->(w)
        SET r.pos_weight = row.pos_w, r.pub_year = row.year
        WITH a, w, row
        WHERE row.iid IS NOT NULL
        MATCH (i:Institution {id: row.iid})
        MERGE (a)-[:AFFILIATED_WITH]->(i)
        WITH w, row
        WHERE row.sid IS NOT NULL
        MATCH (s:Source {id: row.sid})
        MERGE (w)-[:PUBLISHED_IN]->(s)
    """,
    "LINK_TAGGED": "UNWIND $data AS row MATCH (w:Work {id: row.wid}), (v:Vocabulary {id: row.vid}) MERGE (w)-[:TAGGED]->(v)",
    "LINK_SIMILAR": "UNWIND $data AS row MATCH (v1:Vocabulary {id: row.f}), (v2:Vocabulary {id: row.t}) MERGE (v1)-[r:SIMILAR_TO]->(v2) SET r.score = row.s"
}