import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class AlphabetImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet Image Viewer")
        self.root.geometry("800x600")
        
        # Folder where images are stored
        self.image_folder = r"C:\Users\Alumno.F10KLAB103PC12\Desktop\LETRAS"
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # List of letters
        self.letter_list = tk.StringVar()
        self.letter_entry = ttk.Entry(self.main_frame, textvariable=self.letter_list)
        self.letter_entry.grid(row=0, column=0, padx=5, pady=5)
        
        # Load button
        self.load_button = ttk.Button(self.main_frame, text="Load Images", command=self.load_images)
        self.load_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.main_frame, width=600, height=400)
        self.canvas.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Status label
        self.status = tk.StringVar()
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status)
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Store image references
        self.image_references = []
        
    def load_images(self):
        # Clear previous images
        self.canvas.delete("all")
        self.image_references.clear()
        
        # Get letters from entry
        letters = self.letter_list.get().replace(" ", "").split(",")
        
        if not letters or letters == ['']:
            self.status.set("Please enter at least one letter")
            return
            
        x_pos = 10
        y_pos = 10
        max_width = 600
        
        valid_extensions = ['.png', '.jpg', '.jpeg', '.gif']
        
        for letter in letters:
            if not letter:
                continue
                
            # Check both lowercase and uppercase
            letter_lower = letter.lower()
            if not letter_lower.isalpha() or len(letter_lower) != 1:
                self.status.set(f"Invalid input: {letter}. Please use single letters")
                return
                
            # Try to find image (case insensitive)
            found = False
            for ext in valid_extensions:
                for case_letter in [letter_lower, letter_lower.upper()]:
                    image_path = os.path.join(self.image_folder, f"{case_letter}{ext}")
                    if os.path.exists(image_path):
                        try:
                            # Load and resize image
                            img = Image.open(image_path)
                            # Calculate aspect ratio
                            aspect = max(100.0/img.width, 100.0/img.height)
                            new_size = (int(img.width*aspect), int(img.height*aspect))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                            
                            # Convert to PhotoImage
                            photo = ImageTk.PhotoImage(img)
                            self.image_references.append(photo)
                            
                            # Display on canvas
                            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=photo)
                            x_pos += new_size[0] + 10
                            
                            # Move to next row if needed
                            if x_pos > max_width - new_size[0]:
                                x_pos = 10
                                y_pos += new_size[1] + 10
                            
                            found = True
                            break
                        except Exception as e:
                            self.status.set(f"Error loading {image_path}: {str(e)}")
                            return
                    if found:
                        break
                        
            if not found:
                self.status.set(f"No image found for letter: {letter}")
                return
                
        self.status.set(f"Loaded images for: {', '.join(letters)}")

def main():
    # Create images folder if it doesn't exist
    if not os.path.exists("alphabet_images"):
        os.makedirs("alphabet_images")
        
    root = tk.Tk()
    app = AlphabetImageViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()