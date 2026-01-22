import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import threading


class Config:
    """Configuration class for all paths and settings"""
    DATASET_PATH = "dataset"
    MODELS_PATH = "models"
    ATTENDANCE_PATH = "Attendance"
    ENCODINGS_FILE = os.path.join(MODELS_PATH, "known_encodings.npy")
    NAMES_FILE = os.path.join(MODELS_PATH, "known_names.npy")
    TOLERANCE = 0.6  
    SCALE_FACTOR = 0.25

def setup_directories():
    """Create necessary directories if they don't exist"""
    for path in [Config.DATASET_PATH, Config.MODELS_PATH, Config.ATTENDANCE_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"✓ Created directory: {path}")


class DatasetCollector:
    """Handles dataset collection using webcam"""
    
    def __init__(self):
        self.cap = None
        self.person_name = None
        self.images_captured = 0
        self.target_images = 50  
    
    def start_collection(self, person_name):
        """Start capturing images for a person"""
        if not person_name or person_name.strip() == "":
            messagebox.showerror("Error", "Please enter a valid name!")
            return False
        
        self.person_name = person_name.strip()
        self.images_captured = 0
        
        # Create person's folder
        person_folder = os.path.join(Config.DATASET_PATH, self.person_name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera!")
            return False
        
        print(f"\n📸 Starting dataset collection for: {self.person_name}")
        print(f"Target: {self.target_images} images")
        print("Press 'SPACE' to capture, 'Q' to quit early")
        
        while self.images_captured < self.target_images:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read from camera!")
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Captured: {self.images_captured}/{self.target_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Person: {self.person_name}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, Q to quit", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Dataset Collection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key
                # Save image
                img_path = os.path.join(person_folder, f"{self.person_name}_{self.images_captured}.jpg")
                cv2.imwrite(img_path, frame)
                self.images_captured += 1
                print(f"✓ Captured image {self.images_captured}/{self.target_images}")
            elif key == ord('q'):  # Quit early
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.images_captured > 0:
            messagebox.showinfo("Success", 
                              f"✓ Captured {self.images_captured} images for {self.person_name}!")
            return True
        else:
            messagebox.showwarning("Warning", "No images captured!")
            return False


class ModelTrainer:
    """Handles face encoding and model training"""
    
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
    
    def train_model(self):
        """Train the model by encoding all faces in dataset"""
        print("\n🔄 Starting model training...")
        
        # Check if dataset exists
        if not os.path.exists(Config.DATASET_PATH):
            messagebox.showerror("Error", "Dataset folder not found!")
            return False
        
        person_folders = [f for f in os.listdir(Config.DATASET_PATH) 
                         if os.path.isdir(os.path.join(Config.DATASET_PATH, f))]
        
        if len(person_folders) == 0:
            messagebox.showerror("Error", "No person folders found in dataset!")
            return False
        
        total_images = 0
        successful_encodings = 0
        
        # Process each person's folder
        for person_name in person_folders:
            person_path = os.path.join(Config.DATASET_PATH, person_name)
            image_files = [f for f in os.listdir(person_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\n📂 Processing {person_name}: {len(image_files)} images")
            
            for img_file in image_files:
                total_images += 1
                img_path = os.path.join(person_path, img_file)
                
                try:
                    # Load image
                    image = face_recognition.load_image_file(img_path)
                    
                    # Get face encodings
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        # Use the first face found
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(person_name)
                        successful_encodings += 1
                    else:
                        print(f"  ⚠ No face detected in {img_file}")
                
                except Exception as e:
                    print(f"  ✗ Error processing {img_file}: {str(e)}")
        
        # Save encodings
        if successful_encodings > 0:
            np.save(Config.ENCODINGS_FILE, self.known_encodings)
            np.save(Config.NAMES_FILE, self.known_names)
            
            print(f"\n✓ Training complete!")
            print(f"  Total images processed: {total_images}")
            print(f"  Successful encodings: {successful_encodings}")
            print(f"  Unique persons: {len(set(self.known_names))}")
            
            messagebox.showinfo("Success", 
                              f"✓ Model trained successfully!\n\n"
                              f"Total encodings: {successful_encodings}\n"
                              f"Persons: {len(set(self.known_names))}")
            return True
        else:
            messagebox.showerror("Error", "No faces were successfully encoded!")
            return False


class AttendanceSystem:
    """Handles real-time face recognition and attendance marking"""
    
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.marked_today = set()  
        self.cap = None
        self.is_running = False
    
    def load_encodings(self):
        """Load trained encodings"""
        try:
            if not os.path.exists(Config.ENCODINGS_FILE) or not os.path.exists(Config.NAMES_FILE):
                messagebox.showerror("Error", "Model not trained yet! Please train the model first.")
                return False
            
            self.known_encodings = np.load(Config.ENCODINGS_FILE, allow_pickle=True)
            self.known_names = np.load(Config.NAMES_FILE, allow_pickle=True)
            
            print(f"\n✓ Loaded {len(self.known_encodings)} encodings")
            print(f"✓ Unique persons: {len(set(self.known_names))}")
            return True
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load encodings: {str(e)}")
            return False
    
    def mark_attendance(self, name):
        """Mark attendance in Excel file"""
        if name == "Unknown" or name in self.marked_today:
            return
        
        # Get current date and time
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Create attendance file path
        attendance_file = os.path.join(Config.ATTENDANCE_PATH, f"Attendance_{date_str}.xlsx")
        
        # Prepare data
        attendance_data = {
            'Name': [name],
            'Date': [date_str],
            'Time': [time_str]
        }
        
        # Save to Excel
        try:
            if os.path.exists(attendance_file):
                # Append to existing file
                existing_df = pd.read_excel(attendance_file)
                new_df = pd.DataFrame(attendance_data)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(attendance_file, index=False)
            else:
                # Create new file
                df = pd.DataFrame(attendance_data)
                df.to_excel(attendance_file, index=False)
            
            self.marked_today.add(name)
            print(f"✓ Attendance marked for {name} at {time_str}")
        
        except Exception as e:
            print(f"✗ Error marking attendance: {str(e)}")
    
    def start_recognition(self):
        """Start real-time face recognition"""
        if not self.load_encodings():
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera!")
            return
        
        self.is_running = True
        print("\n🎥 Starting face recognition...")
        print("Press 'Q' to quit")
        
        # Reset marked today list
        self.marked_today.clear()
        
        frame_count = 0
        process_every_n_frames = 3  # Process every 3rd frame for better performance
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % process_every_n_frames == 0:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=Config.SCALE_FACTOR, fy=Config.SCALE_FACTOR)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Process each face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back coordinates
                    top = int(top / Config.SCALE_FACTOR)
                    right = int(right / Config.SCALE_FACTOR)
                    bottom = int(bottom / Config.SCALE_FACTOR)
                    left = int(left / Config.SCALE_FACTOR)
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, 
                                                            tolerance=Config.TOLERANCE)
                    name = "Unknown"
                    confidence = 0
                    
                    if True in matches:
                        # Get face distances
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.known_names[best_match_index]
                            confidence = (1 - face_distances[best_match_index]) * 100
                            
                            # Mark attendance for known faces
                            self.mark_attendance(name)
                    
                    # Draw rectangle and label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    # Display name and confidence
                    label = f"{name}"
                    if name != "Unknown":
                        label += f" ({confidence:.1f}%)"
                        if name in self.marked_today:
                            label += " ✓"
                    
                    cv2.putText(frame, label, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display attendance count
            cv2.putText(frame, f"Attendance Marked: {len(self.marked_today)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Face Recognition Attendance", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.is_running = False
        
        if len(self.marked_today) > 0:
            messagebox.showinfo("Session Complete", 
                              f"✓ Attendance marked for {len(self.marked_today)} person(s):\n" + 
                              "\n".join(sorted(self.marked_today)))
        else:
            messagebox.showinfo("Session Complete", "No attendance was marked.")


class AttendanceGUI:
    """Main GUI application using Tkinter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Initialize modules
        self.dataset_collector = DatasetCollector()
        self.model_trainer = ModelTrainer()
        self.attendance_system = AttendanceSystem()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Face Recognition Attendance System", 
                              font=("Arial", 18, "bold"), fg="#2c3e50")
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, text="Automated Attendance Management", 
                                 font=("Arial", 10), fg="#7f8c8d")
        subtitle_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=30)
        
        # Button style
        btn_width = 25
        btn_height = 2
        btn_font = ("Arial", 11, "bold")
        
        # Capture Dataset Button
        self.btn_capture = tk.Button(button_frame, text="📸 Capture Dataset", 
                                     command=self.capture_dataset,
                                     width=btn_width, height=btn_height, font=btn_font,
                                     bg="#3498db", fg="white", cursor="hand2")
        self.btn_capture.pack(pady=8)
        
        # Train Model Button
        self.btn_train = tk.Button(button_frame, text="🧠 Train Model", 
                                   command=self.train_model,
                                   width=btn_width, height=btn_height, font=btn_font,
                                   bg="#2ecc71", fg="white", cursor="hand2")
        self.btn_train.pack(pady=8)
        
        # Start Attendance Button
        self.btn_attendance = tk.Button(button_frame, text="✓ Start Attendance", 
                                       command=self.start_attendance,
                                       width=btn_width, height=btn_height, font=btn_font,
                                       bg="#e74c3c", fg="white", cursor="hand2")
        self.btn_attendance.pack(pady=8)
        
        # Exit Button
        self.btn_exit = tk.Button(button_frame, text="🚪 Exit", 
                                 command=self.exit_app,
                                 width=btn_width, height=btn_height, font=btn_font,
                                 bg="#95a5a6", fg="white", cursor="hand2")
        self.btn_exit.pack(pady=8)
        
        # Footer
        footer_label = tk.Label(self.root, text="© 2025 - Face Recognition System", 
                               font=("Arial", 8), fg="#bdc3c7")
        footer_label.pack(side=tk.BOTTOM, pady=10)
    
    def capture_dataset(self):
        """Handle capture dataset button click"""
        name = simpledialog.askstring("Input", "Enter person's name:", parent=self.root)
        if name:
            # Run in separate thread to prevent GUI freeze
            thread = threading.Thread(target=self.dataset_collector.start_collection, args=(name,))
            thread.start()
    
    def train_model(self):
        """Handle train model button click"""
        # Run in separate thread
        thread = threading.Thread(target=self.model_trainer.train_model)
        thread.start()
    
    def start_attendance(self):
        """Handle start attendance button click"""
        # Run in separate thread
        thread = threading.Thread(target=self.attendance_system.start_recognition)
        thread.start()
    
    def exit_app(self):
        """Handle exit button click"""
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            self.root.destroy()


def main():
    """Main function to run the application"""
    # Setup directories
    setup_directories()
    
    # Create and run GUI
    root = tk.Tk()
    app = AttendanceGUI(root)
    
    print("=" * 60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 60)
    print("\n🚀 Application started successfully!")
    print("✓ Use the GUI to interact with the system\n")
    
    root.mainloop()

if __name__ == "__main__":
    main()