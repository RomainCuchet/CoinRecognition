import os
import shutil

from config import Config

def rename_pictures(folder_path, prefix):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    files.sort()
    
    for i, filename in enumerate(files):
        file_extension = os.path.splitext(filename)[1]
        
        new_filename = f"{prefix}{i}{file_extension}"
        
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        os.rename(old_file_path, new_file_path)
    
    print(f"Renamed {len(files)} files successfully.")
    
def organize_files(src_folder):
    images_folder = os.path.join(src_folder, 'images')
    annotations_folder = os.path.join(src_folder, 'annotations')
    
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)
    
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        
        if os.path.isfile(file_path):
            if filename.lower().endswith('.jpg'):
                shutil.move(file_path, images_folder)
                
            elif filename.lower().endswith('.xml'):
                shutil.move(file_path, annotations_folder)
                
import os
import xml.etree.ElementTree as ET

def extract_unique_objects(folder_path, labels=list(Config().LABELS.keys())):
    object_names = {}
    for filename in os.listdir(folder_path):
        n = 0
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for obj in root.findall(".//object"):
                n += 1
                name = obj.find("name").text
                if name in labels:
                    if name in object_names:
                        object_names[name] += 1
                    else:
                        object_names[name] = 1
                
        s = 0
        for name in object_names:
            s += object_names[name]

    return object_names, s

def extract_unique_object(path):
    object_names = {}
    if path.endswith(".xml"):
        
        tree = ET.parse(path)
        root = tree.getroot()
        
        for obj in root.findall(".//object"):
            name = obj.find("name").text
            if name in object_names:
                object_names[name] += 1
            else:
                object_names[name] = 1
    return object_names

def count_files_in_directory(directory):
    try:
        # Liste tous les fichiers et dossiers dans le répertoire donné
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return len(files)
    except FileNotFoundError:
        print("Le répertoire spécifié n'existe pas.")
        return 0
    
def delete_no_labels(image_folder, annotation_folder,labels:dict[str:int]):
    n=0
    for xml_file in os.listdir(annotation_folder):
        xml_path = os.path.join(annotation_folder, xml_file)
        
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Check if the XML has any 'object' tags
        objects = root.findall("object")
        if not objects :  # No objects found
            n=delete_file(image_folder, n, xml_file, xml_path, objects)
        else:
            verif = False
            for obj in objects:
                label = obj.find("name").text
                if label in labels.keys():
                    verif = True
            if not verif : n=delete_file(image_folder, n, xml_file, xml_path, objects)
    print(f"{n} file deleted succesfully")

def delete_file(image_folder, n, xml_file, xml_path, objects):
    # Delete the XML file
    os.remove(xml_path)
        
        # Corresponding image filename (assuming .jpg format)
    image_file = xml_file.replace(".xml", ".jpg")
    image_path = os.path.join(image_folder, image_file)
        
        # Check if the corresponding image exists and delete it
    if os.path.exists(image_path):
        os.remove(image_path)
        n+=1
    return n

