# 💵 Fake Currency Detector Using AI

### AI-Powered Indian Currency Denomination Recognition System

An AI-powered web application that analyzes images of Indian currency notes and predicts their denomination using a deep learning model. Users can upload a currency image or capture one directly through their camera.

**Note:** The current model recognizes currency denominations. Genuine-versus-counterfeit verification is not yet reliably implemented.

<p align="center">
  <a href="https://fake-currency-detector-22rw.onrender.com/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20Application-00C853?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/nishad-uday/fake-currency-detector">
    <img src="https://img.shields.io/badge/GitHub-Source%20Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

---

## 🌐 Live Demo

**Try the application:**  
https://fake-currency-detector-22rw.onrender.com/

The application is deployed on Render. On the free hosting plan, it may take some time to respond after a period of inactivity.

## 📸 Application Screenshots

### 1. Home Page

The dashboard provides a clean interface for uploading currency images, capturing notes using a camera, and viewing prediction results.

![Home Page](screenshots/home.png)

### 2. Currency Prediction Result

The application displays the predicted denomination, denomination model score, and basic image-quality checks.

![Prediction Result](screenshots/prediction.png)

---

## ✨ Features

- **AI-Based Denomination Recognition:** Predicts the denomination shown in an Indian currency image.
- **Seven Denominations:** Supports ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 classes.
- **Image Upload:** Upload currency images in JPG, JPEG, PNG, and WEBP formats.
- **Live Camera Capture:** Capture a currency note directly using a supported camera.
- **Prediction Score:** Displays the model's score for the predicted denomination.
- **Image Preview:** Preview the selected image before analysis.
- **Image-Quality Checks:** Provides basic warnings for potentially unsuitable images.
- **Input Safeguards:** Applies file-size and image-dimension checks to reduce unsafe or resource-intensive image processing.
- **Responsive Interface:** Dark-themed dashboard designed for desktop and mobile screens.
- **Web-Based Application:** Use the application through a browser without manually running the model interface.
- **Online Deployment:** Access the application through its public Render URL.

## 🧠 How It Works

The application follows this workflow:

1. **Input:** Upload a currency image or capture one using the camera.
2. **Validation:** Check the image format, size, and basic image properties.
3. **Preprocessing:** Prepare the image for the deep learning model.
4. **Prediction:** Use a ResNet18-based model to predict the denomination.
5. **Display:** Show the predicted denomination, model score, and image-quality information.
6. **Authenticity Status:** Clearly indicate that genuine-versus-counterfeit verification is not currently available.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Backend web framework |
| PyTorch | Deep learning framework |
| Torchvision | Image transformations and model utilities |
| ResNet18 | Currency denomination classification |
| Pillow (PIL) | Image loading and preprocessing |
| OpenCV | Computer vision utilities |
| HTML5 | Web page structure |
| CSS3 | Styling and responsive design |
| JavaScript | Browser interactions and camera workflow |
| Gunicorn | Production WSGI server |
| Git and GitHub | Version control and source code hosting |
| Render | Web application deployment |

## 📂 Project Structure

```text
fake-currency-detector/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── currency_resnet18.pth
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│
├── screenshots/
│   ├── home.png
│   └── prediction.png
│
├── src/
│   └── Dataset, training and evaluation utilities
│
└── .venv/
```

**Important:** This is a simplified representation of the project. The `.venv/` folder is generated locally and should not be uploaded to GitHub. Dataset folders are also excluded from version control where configured in `.gitignore`.

## ⚙️ Installation and Setup

Follow these steps to run the project on your local computer.

### Prerequisites

Install the following:

- Python compatible with the project's dependencies
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

Open PowerShell or the VS Code terminal and run:

```bash
git clone https://github.com/nishad-uday/fake-currency-detector.git
```

Navigate into the project:

```bash
cd fake-currency-detector
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can activate the environment from Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The PyTorch CPU package configuration in `requirements.txt` is intended for CPU-based deployment and does not require an NVIDIA GPU.

### Step 4: Check the Model File

Ensure that the trained denomination model exists at:

```text
models/currency_resnet18.pth
```

If the checkpoint is missing from the repository, obtain the correct trained checkpoint before running the application. The model file must match the architecture and class mapping expected by `app.py`.

### Step 5: Run the Application

```bash
python app.py
```

Open your browser and visit:

http://127.0.0.1:5000/

Upload an image or use the camera to test denomination recognition.

## 🖥️ Using the Application

1. Open the application in your browser.
2. Select a currency image from your device.
3. Preview the selected image.
4. Click **Analyze Uploaded Image**.
5. Alternatively, start the camera, capture a note, and analyze the captured image.
6. Review the predicted denomination and model score.
7. Read any image-quality warnings and the authenticity status.

For more useful predictions, use a clear image with good lighting, the note fully visible, and minimal blur or obstruction.

## 📊 Model Information

The application uses a ResNet18-based image classification model implemented with PyTorch.

The model is configured to predict these seven denomination classes:

- ₹10
- ₹20
- ₹50
- ₹100
- ₹200
- ₹500
- ₹2000

The model score shown in the application relates to **denomination prediction**, not to the probability that a banknote is genuine.

A high score does not guarantee that the predicted denomination is correct, and it does not indicate that the note is genuine.

## ⚠️ Current Limitations

This project is currently a **currency denomination recognition system**, not a validated counterfeit-detection system.

- Genuine-versus-counterfeit verification is not currently available.
- The experimental authenticity classifier uses unverified labels and must not be treated as reliable evidence.
- Reference-based authenticity screening is not available when suitable verified reference data is missing.
- Basic image-quality checks do not verify security features.
- Prediction results can be affected by image quality, lighting, orientation, and differences between training images and real-world images.
- The displayed model score is not a guarantee of correctness or authenticity.

**For financial or official decisions, do not rely on this application to determine whether a banknote is genuine.** Use official Reserve Bank of India guidance and appropriate verification procedures.

## 🚀 Future Improvements

Potential future development includes:

- Building a reliable genuine-versus-counterfeit classifier using properly verified and representative data.
- Collecting and validating genuine and counterfeit image labels.
- Evaluating the model on independent real-world images.
- Investigating relevant banknote security features.
- Improving robustness to lighting, blur, rotation, and different camera devices.
- Optimizing inference memory usage for limited-resource hosting.
- Adding automated tests and model evaluation reports.
- Providing clearer model limitations and confidence calibration.

These are planned improvements and are not claims about existing functionality.

## 🔒 Privacy and Responsible Use

- Upload only images you have permission to use.
- Avoid uploading personal or sensitive information.
- Do not use model predictions as the sole basis for financial decisions.
- Review the deployment's data-handling and retention practices before uploading sensitive images.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork this repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit your changes with a descriptive message.
5. Open a pull request explaining your improvements.

## 👨‍💻 Developer

**GitHub:** [nishad-uday](https://github.com/nishad-uday)

**Repository:** [Fake Currency Detector Using AI](https://github.com/nishad-uday/fake-currency-detector)

## 📄 License

No license is specified in this README. Unless a license is added to the repository, reuse and redistribution are subject to the applicable copyright rules.

---

<p align="center">
  Built with Python, PyTorch, Flask, and computer vision.
</p>

<p align="center">
  <b>Fake Currency Detector Using AI</b>
</p>
