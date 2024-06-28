import os
import pprint
import sqlite3


connection = sqlite3.connect('database_tags_example.db')
cursor = connection.cursor()

cursor.execute("""DROP TABLE IF EXISTS object_table;""")
cursor.execute("""CREATE TABLE IF NOT EXISTS object_table (id INTEGER PRIMARY KEY, object_name);""")
cursor.execute("""DROP TABLE IF EXISTS tag_table;""")
cursor.execute("""CREATE TABLE IF NOT EXISTS tag_table (id INTEGER PRIMARY KEY, tag_name);""")
connection.commit()

cursor.execute("""INSERT INTO object_table (object_name) VALUES ('Object1'),('Object2'),('Object3'),('Object4');""")
cursor.execute("""INSERT INTO tag_table (tag_name) VALUES 
    ('Apple'),('Orange'),('Grape'),('Pineapple'),('Melon'), ('London'),('New York'),('Paris'), ('Red'),('Green'),('Blue'); """)
connection.commit()

cursor.execute("""DROP TABLE IF EXISTS object_tag_mapping;""")
cursor.execute("""CREATE TABLE IF NOT EXISTS object_tag_mapping (object_reference INTEGER, tag_reference INTEGER);""")


cursor.execute("""SELECT * FROM object_table""")
print(cursor.fetchall())

cursor.execute("""SELECT * FROM tag_table""")
print(cursor.fetchall())

cursor.execute("""INSERT INTO object_tag_mapping VALUES
    (1,4), -- obj1 has tag Pineapple
    (1,1),  -- obj1 has Apple
    (1,8), -- obj1 has Paris
    (1,10), -- obj1 has green
    (4,1),(4,3),(4,11), -- some tags for object 4
    (2,8),(2,7),(2,4), -- some tags for object 2
    (3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11);""")
connection.commit()

cursor.execute("""SELECT * FROM object_tag_mapping""")
print(cursor.fetchall())

cursor.execute("""SELECT object_name,  group_concat(tag_name,',') AS tags_for_this_object 
    FROM object_tag_mapping 
    JOIN object_table ON object_reference = object_table.id
    JOIN tag_table ON tag_reference = tag_table.id
    GROUP BY object_name;
""")
connection.commit()

print(cursor.fetchall())
