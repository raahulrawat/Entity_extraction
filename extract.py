from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


class entity:

    def __init__(self):
        # directory for storing the new trained model..
        self.output_dir = "/home/rahul/Downloads/MANET/vocab/"

    def default(self, text):

        """
        :param text: input text from which entities will be extracted.
        :return: will return a json which contains the entities text and label
        """

        nlp = spacy.load('en')
        data_sample = nlp(text)
        entities = {}
        for ent in data_sample.ents:
            entities[ent.label_] = ent.text
        return entities

    def custom(self, label, data, model=None, new_model_name='animal', n_iter=10):
        """
        :param label: input label name for which entity will be trained...
        :param data: training dataset to train custom model..
        :param model: provide model None or standard module to train...
        :param new_model_name: give name to create a new model..
        :param n_iter: number of iterations used to train the model..
        :return: status of the process..
        """
        """Set up the pipeline and entity recognizer, and train the new entity."""
        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank('en')  # create blank Language class
            print("Created blank 'en' model")

        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)

        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe('ner')

        ner.add_label(label)  # add new entity label to entity recognizer
        if model is None:
            optimizer = nlp.begin_training()
        else:
            # Note that 'begin_training' initializes the models, so it'll zero out
            # existing entity types.
            optimizer = nlp.entity.create_optimizer()

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(n_iter):
                random.shuffle(data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(data, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                               losses=losses)
                print('Losses', losses)

        # save model to output directory
        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.meta['name'] = new_model_name  # rename model
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)
        isSuccess = True
        return isSuccess

    def load_model(self, test_text):
        """
        :param test_text: text to test the trained model..
        :return: will return the detected entities..
        """
        # test the saved model
        print("Loading from", self.output_dir)
        nlp2 = spacy.load(self.output_dir)
        doc2 = nlp2(test_text)
        entities = []
        for ent in doc2.ents:
            print(ent.label_, ent.text)
            entities[ent.label_] = ent.text
        return entities
