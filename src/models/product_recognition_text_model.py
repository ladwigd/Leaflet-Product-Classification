# +
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import torch
import threading

#preprocessoing
from os import listdir
import cv2 
import pytesseract
from pytesseract import Output
import time
import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# +
class productRecognitionTextModel:
    def __init__(self):
        self.vect_vocab = None
        
    def train_vocab(self, df_series):
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.vect_vocab=vectorizer.fit(df_series)
    
    def transform(self, df_series):
        return self.vect_vocab.transform(df_series)
    
    def train(self, train_transformed, truth_labels_series):  
        self.model = SGDClassifier(random_state = 0, loss="modified_huber", learning_rate = "adaptive", eta0=1)
        self.model = self.model.fit(train_transformed, truth_labels_series)
        return self.model
    
    def predict(self, input_transformed):
        return self.model.predict(input_transformed)
       
class ocrExtraction:
    """
       Preprocess helper 
    """
    def ocr_extraction(self, extract_image, psm_level = 3):
        custom_config = r'--oem 3 --psm ' + str(psm_level)
        return pytesseract.image_to_string(extract_image, lang="deu", config = custom_config)
    
    def resize_preprocess(self, preprocess_image):
        return cv2.resize(preprocess_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    def gray_preprocess(self, preprocess_image):
        return cv2.cvtColor(preprocess_image, cv2.COLOR_BGR2GRAY)

    def threshold_preprocess(self, preprocess_image, thresh = 0, maxval = 255):
        return cv2.threshold(preprocess_image, thresh, maxval, cv2.THRESH_OTSU)[1]
    
    """
        Concrete extraction functions
    """
    def ocr_extract_3psm(self, extract_image):
        return self.ocr_extraction(extract_image, psm_level = 3)

    def ocr_extract_6psm(self, extract_image):
        return self.ocr_extraction(extract_image, psm_level = 6)

    def ocr_extract_11psm(self, extract_image):
        return self.ocr_extraction(extract_image, psm_level = 11)

    def ocr_extract_12psm(self, extract_image):
        return self.ocr_extraction(extract_image, psm_level = 12)

    def ocr_extract_gray_psm3(self, extract_image):
        gray = self.gray_preprocess(extract_image)
        return self.ocr_extraction(gray, psm_level = 3)

    def ocr_extract_treshold(self, extract_image):
        resized = self.resize_preprocess(extract_image)
        gray = self.gray_preprocess(resized)
        thresh = self.threshold_preprocess(gray, thresh = 0, maxval = 255)
        return self.ocr_extract_3psm(thresh)

    def ocr_extract_gray_resize_psm11(self, extract_image):
        resized = self.resize_preprocess(extract_image)
        gray = self.gray_preprocess(resized)
        return self.ocr_extraction(gray, psm_level = 11)

    def ocr_extract_gray_resize_psm6(self, extract_image):
        resized = self.resize_preprocess(extract_image)
        gray = self.gray_preprocess(resized)
        return self.ocr_extraction(gray, psm_level = 6)
    
    def start_ocr_extraction_by_path(self, extraction_handler, parse_path, save_path, save_filename):
        df_extracted = pd.DataFrame()
        track_time_start = time.time()
        print("Start - time: ", datetime.datetime.now())

        parse_path = parse_path 
        ocr_strings = []
        image_paths = []
        classes_test_folder = listdir(parse_path)
        truth_classes = []

        for count, i in enumerate(classes_test_folder):
            image_names = listdir(parse_path + str(i))
            for i_name in image_names:
                extract_image_path = parse_path + str(i) + "/" + str(i_name)
                img = mpimg.imread(extract_image_path)

                parsed_string = extraction_handler(img)

                ocr_strings.append(parsed_string)
                image_paths.append(extract_image_path)
                truth_classes.append(i)
            if count == 5:
                track_time_end = time.time()
                print("Predicted extraction Time", (((track_time_end - track_time_start)/5) * len(classes_test_folder))/60)
            if count % 10:
                print(str(count) + "/" + str(len(classes_test_folder)))

        track_time_end = time.time()
        extract_duration = track_time_end - track_time_start

        df_extracted["truth_classes"] = truth_classes
        df_extracted["ocr_strings"] = ocr_strings
        df_extracted["image_paths"] = image_paths

        df_extracted.to_pickle(save_path + save_filename + ".pkl")
        with open(save_path + save_filename + "_extraction_time.pkl", 'wb') as file:
            pickle.dump(extract_duration, file)
        print("End - time: ", datetime.datetime.now())
        print("Extraction Duration: " + str(extract_duration/60) + " minutes")
        
    def start_ocr_extraction_by_df(self, extraction_handler, input_df, save_path, save_filename):
        df_extracted = pd.DataFrame()
        track_time_start = time.time()
        ocr_strings = []

        for count, i in enumerate(input_df):
            image_names = input_df["image_paths"].values
            img = mpimg.imread(image_names[count])
            parsed_string = extraction_handler(img)
            ocr_strings.append(parsed_string)
        df_extracted["ocr_strings"] = ocr_strings
        return df_extracted
            
    extraction_methods = [ocr_extract_3psm, ocr_extract_6psm, ocr_extract_11psm, ocr_extract_12psm, 
                          ocr_extract_gray_psm3, ocr_extract_treshold, ocr_extract_gray_resize_psm11
                         , ocr_extract_gray_resize_psm6]
    
    def ocr_image(self, img, return_single_results = False, multi_threaded=True):
        extraction_string = ""
        extracted_strings = []
        threads = []
        if multi_threaded == True:
            for count, ex_method in enumerate(self.extraction_methods):
                def thread_extraction(ex_method):
                    extracted = getattr(self, str(ex_method.__name__))(img)
                    extracted_strings.append(extracted)
                t = threading.Thread(target=thread_extraction, args=[ex_method])
                threads.append(t)
            for x in threads:
                x.start()
            for x in threads:
                x.join()
        else:
            for count, ex_method in enumerate(self.extraction_methods):
                extracted = getattr(self, str(ex_method.__name__))(img)
                extracted_strings.append(extracted)
        if return_single_results == True:
            return extracted_strings
        else:
            for i in extracted_strings:
                extraction_string = extraction_string + " " + i
            return extraction_string
