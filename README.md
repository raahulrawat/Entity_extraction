# Entity_extraction

Standard pretrained based Entity extraction and custom Entity extraction using SpaCy.
Entities can be extracted from text using SpaCy are:
```
                        Type	        Description
                        PERSON          People, including fictional.
                        NORP	        Nationalities or religious or political groups.
                        FAC	            Buildings, airports, highways, bridges, etc.
                        ORG	            Companies, agencies, institutions, etc.
                        GPE	            Countries, cities, states.
                        LOC	            Non-GPE locations, mountain ranges, bodies of water.
                        PRODUCT	        Objects, vehicles, foods, etc. (Not services.)
                        EVENT	        Named hurricanes, battles, wars, sports events, etc.
                        WORK_OF_ART	    Titles of books, songs, etc.
                        LAW          	Named documents made into laws.
                        LANGUAGE	    Any named language.
                        DATE	        Absolute or relative dates or periods.
                        TIME	        Times smaller than a day.
                        PERCENT	        Percentage, including "%".
                        MONEY	        Monetary values, including unit.
                        QUANTITY	    Measurements, as of weight or distance.
                        ORDINAL	        "first", "second", etc.
                        CARDINAL	    Numerals that do not fall under another type.
```

Requirment:
spaCy
pathlib

Reference:  https://spacy.io/usage/training/#section-ner
usage
```
sample training data

[
    ("Horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("Do they bite?", {
        'entities': []
    }),

    ("horses are too tall and they pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("horses pretend to care about your feelings", {
        'entities': [(0, 6, 'ANIMAL')]
    }),

    ("they pretend to care about your feelings, those horses", {
        'entities': [(48, 54, 'ANIMAL')]
    }),

    ("horses?", {
        'entities': [(0, 6, 'ANIMAL')]
    })
]

use atleast 100 samples of data for each lable to get accuracy from the model.
```

``` python

from extract import entity
entity = entity()

To get standard entities results
text = "sample text"
ent = entity.default(text)
print(ent)

training model
t_data = "training sample data"
label = "sample label"
ent = entity.trainModel(label= label, data = t_data)

testing model
text = "sample text"
ent =  entity.load_model(test_text = text)
print(ent)

```
