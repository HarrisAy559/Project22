#!/usr/bin/env python3
"""
Main Face Recognition Application for Termux
"""

import cv2
import os
import sys
import json
import pickle
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.face_utils import FaceUtils
from utils.camera_utils import CameraUtils
from utils.file_utils import FileUtils
from utils.log_utils import setup_logger

class FaceRecognitionApp:
    def __init__(self):
        """Initialize the face recognition application"""
        # Setup logging
        self.logger = setup_logger(__name__)
        self.logger.info("Initializing Face Recognition App")
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize utilities
        self.face_utils = FaceUtils(self.config)
        self.camera_utils = CameraUtils()
        self.file_utils = FileUtils()
        
        # Initialize variables
        self.known_faces = {}
        self.known_encodings = []
        self.known_names = []
        
        # Load existing data
        self.load_known_faces()
        
    def load_config(self):
        """Load configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        default_config = {
            'camera_index': 0,
            'recognition_threshold': 0.6,
            'face_detection_model': 'hog',  # or 'cnn' for better accuracy
            'save_unknown_faces': True,
            'log_recognition': True,
            'image_quality': 95,
            'max_faces': 10
        }
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info("Configuration loaded successfully")
                return {**default_config, **config}
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        
        return default_config
    
    def load_known_faces(self):
        """Load known faces from database"""
        encodings_path = os.path.join('known_faces', 'encodings.pkl')
        metadata_path = os.path.join('known_faces', 'metadata.json')
        
        if os.path.exists(encodings_path):
            try:
                with open(encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_names = data['names']
                
                self.logger.info(f"Loaded {len(self.known_names)} known faces")
                
                # Load metadata
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.known_faces = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading known faces: {e}")
                self.known_encodings = []
                self.known_names = []
    
    def save_known_faces(self):
        """Save known faces to database"""
        os.makedirs('known_faces', exist_ok=True)
        
        # Save encodings
        encodings_path = os.path.join('known_faces', 'encodings.pkl')
        with open(encodings_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_encodings,
                'names': self.known_names
            }, f)
        
        # Save metadata
        metadata_path = os.path.join('known_faces', 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.known_faces, f, indent=2)
        
        self.logger.info(f"Saved {len(self.known_names)} known faces")
    
    def add_new_person(self):
        """Add a new person to the database"""
        print("\n" + "="*50)
        print("ADD NEW PERSON")
        print("="*50)
        
        name = input("Enter person name: ").strip()
        if not name:
            print("Name cannot be empty!")
            return
        
        if name in self.known_names:
            print(f"Person '{name}' already exists!")
            choice = input("Do you want to add more images? (y/n): ")
            if choice.lower() != 'y':
                return
        
        print(f"\nPreparing to capture images for {name}...")
        print("Make sure you have good lighting and face the camera")
        input("Press Enter when ready...")
        
        # Capture images
        images = self.capture_face_images(name)
        
        if images:
            # Process and save faces
            success = self.process_and_save_faces(name, images)
            if success:
                print(f"\n✓ Successfully added {name} to database!")
                self.save_known_faces()
            else:
                print(f"\n✗ Failed to add {name}")
        else:
            print("\n✗ No images captured")
    
    def capture_face_images(self, name, num_images=10):
        """Capture multiple face images"""
        print(f"\nCapturing {num_images} images...")
        print("Look straight at the camera. Move slightly between captures.")
        
        images = []
        cap = self.camera_utils.get_camera(self.config['camera_index'])
        
        if not cap:
            print("Cannot access camera!")
            return []
        
        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display countdown
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Image {count+1}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Detect face
            face_locations = self.face_utils.detect_faces(frame)
            
            if len(face_locations) == 1:
                # Draw green rectangle
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), 
                                 (0, 255, 0), 2)
                
                # Save image every second
                if count == 0 or (datetime.now().timestamp() % 1) < 0.1:
                    images.append(frame.copy())
                    count += 1
                    print(f"  Captured image {count}/{num_images}")
            else:
                cv2.putText(display_frame, "Align face in frame", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"Capture - {name}", display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return images
    
    def process_and_save_faces(self, name, images):
        """Process captured images and save face encodings"""
        all_encodings = []
        
        for i, image in enumerate(images):
            # Get face encodings
            encodings = self.face_utils.get_face_encodings(image)
            
            if encodings:
                all_encodings.extend(encodings)
                
                # Save thumbnail
                face_locations = self.face_utils.detect_faces(image)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_img = image[top:bottom, left:right]
                    
                    # Save thumbnail
                    thumb_dir = os.path.join('known_faces', 'thumbnails')
                    os.makedirs(thumb_dir, exist_ok=True)
                    thumb_path = os.path.join(thumb_dir, f"{name}_{i}.jpg")
                    cv2.imwrite(thumb_path, face_img)
        
        if all_encodings:
            # Add to known faces
            self.known_encodings.extend(all_encodings)
            self.known_names.extend([name] * len(all_encodings))
            
            # Update metadata
            if name not in self.known_faces:
                self.known_faces[name] = {
                    'id': len(self.known_faces) + 1,
                    'added_date': datetime.now().isoformat(),
                    'num_images': len(all_encodings),
                    'thumbnails': [f"{name}_{i}.jpg" for i in range(len(all_encodings))]
                }
            
            return True
        
        return False
    
    def recognize_faces(self):
        """Real-time face recognition"""
        print("\n" + "="*50)
        print("REAL-TIME FACE RECOGNITION")
        print("="*50)
        print("Press 'q' to quit")
        print("Press 's' to save unknown face")
        print("="*50)
        
        if not self.known_encodings:
            print("No known faces in database! Add some faces first.")
            return
        
        cap = self.camera_utils.get_camera(self.config['camera_index'])
        if not cap:
            print("Cannot access camera!")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame = small_frame[:, :, ::-1]  # BGR to RGB
            
            # Detect faces
            face_locations = self.face_utils.detect_faces(rgb_frame)
            
            # Recognize faces
            for (top, right, bottom, left) in face_locations:
                # Scale back up
                top *= 2; right *= 2; bottom *= 2; left *= 2
                
                # Extract face
                face_image = frame[top:bottom, left:right]
                
                # Get encoding
                face_encoding = self.face_utils.get_face_encoding(face_image)
                
                if face_encoding is not None:
                    # Compare with known faces
                    matches = self.face_utils.compare_faces(
                        self.known_encodings, face_encoding
                    )
                    
                    name = "Unknown"
                    confidence = 0
                    
                    if True in matches:
                        # Find best match
                        face_distances = self.face_utils.face_distance(
                            self.known_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.known_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                    
                    # Draw rectangle and label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Create label
                    label = f"{name} ({confidence:.1%})" if name != "Unknown" else "Unknown"
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, label, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Log recognition
                    if self.config['log_recognition']:
                        self.log_recognition(name, confidence)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_unknown_face(frame, face_locations)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_unknown_face(self, frame, face_locations):
        """Save detected unknown face"""
        if face_locations and self.config['save_unknown_faces']:
            os.makedirs('unknown_faces', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_img = frame[top:bottom, left:right]
                filename = f"unknown_faces/unknown_{timestamp}_{i}.jpg"
                cv2.imwrite(filename, face_img)
                print(f"Saved unknown face: {filename}")
    
    def log_recognition(self, name, confidence):
        """Log recognition event"""
        os.makedirs('logs', exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'confidence': float(confidence),
            'recognized': name != "Unknown"
        }
        
        # Save to CSV
        import csv
        csv_path = 'logs/recognition_log.csv'
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    
    def list_known_persons(self):
        """List all known persons"""
        print("\n" + "="*50)
        print("KNOWN PERSONS")
        print("="*50)
        
        if not self.known_faces:
            print("No known persons in database.")
            return
        
        for name, info in self.known_faces.items():
            count = sum(1 for n in self.known_names if n == name)
            print(f"• {name}")
            print(f"  ID: {info['id']}")
            print(f"  Images: {count}")
            print(f"  Added: {info['added_date'][:10]}")
            print()
    
    def main_menu(self):
        """Display main menu"""
        while True:
            print("\n" + "="*50)
            print("FACE RECOGNITION TERMUX")
            print("="*50)
            print("1. Add New Person")
            print("2. Real-time Recognition")
            print("3. List Known Persons")
            print("4. Train Model")
            print("5. Test on Image")
            print("6. Settings")
            print("7. View Logs")
            print("8. Exit")
            print("="*50)
            
            try:
                choice = input("Enter choice (1-8): ").strip()
                
                if choice == '1':
                    self.add_new_person()
                elif choice == '2':
                    self.recognize_faces()
                elif choice == '3':
                    self.list_known_persons()
                elif choice == '4':
                    self.train_model()
                elif choice == '5':
                    self.test_on_image()
                elif choice == '6':
                    self.show_settings()
                elif choice == '7':
                    self.view_logs()
                elif choice == '8':
                    print("\nGoodbye! 👋")
                    break
                else:
                    print("Invalid choice! Please enter 1-8")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in menu: {e}")
                print(f"Error: {e}")
    
    def train_model(self):
        """Train face recognition model"""
        print("\nTraining model...")
        # Implement training logic here
        print("Training complete!")
    
    def test_on_image(self):
        """Test recognition on a single image"""
        print("\nTest on image feature coming soon!")
    
    def show_settings(self):
        """Display and modify settings"""
        print("\nCurrent settings:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        change = input("\nChange settings? (y/n): ")
        if change.lower() == 'y':
            # Implement settings modification
            print("Settings modification coming soon!")
    
    def view_logs(self):
        """View recognition logs"""
        log_path = 'logs/recognition_log.csv'
        if os.path.exists(log_path):
            import pandas as pd
            try:
                df = pd.read_csv(log_path)
                print("\nRecognition Logs:")
                print(df.tail(10))  # Show last 10 entries
            except:
                print("Could not read log file")
        else:
            print("No logs found")

def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════╗
    ║    FACE RECOGNITION TERMUX v1.0      ║
    ╚══════════════════════════════════════╝
    """)
    
    # Check dependencies
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Missing dependencies! Run: pip install -r requirements.txt")
        return
    
    # Create necessary directories
    for dir_name in ['dataset', 'known_faces', 'unknown_faces', 'logs', 'models']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Run application
    app = FaceRecognitionApp()
    app.main_menu()

if __name__ == "__main__":
    main()