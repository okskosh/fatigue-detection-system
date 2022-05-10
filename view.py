import tkinter as tki
from detection import drive_process, test_process


def app_view():
    # initialize the root window 
    root = tki.Tk()
    root.title("Fatigue Detection System")
    root.geometry('300x250')

    # button, that when pressed, exits the app
    exit_btn = tki.Button(root, text="Exit", bg="#BDD9BF", command=root.destroy)
    exit_btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

    # button, that when pressed, runs the app in drive mode (with sound alert)
    drive_btn = tki.Button(root, text="Driving mode", bg="#BDD9BF", command=drive_process)
    drive_btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

    # button, that when pressed, runs the app in test mode (no sound alert)
    test_btn = tki.Button(root, text="Testing mode", bg="#BDD9BF", command=test_process)
    test_btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

    # create label on window
    label = tki.Label(root, text="Fatigue Detection", bg="#2E4052", fg='white', height=2)
    label.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)    

    root.mainloop()
