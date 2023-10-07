# Data Preperation code for Bach-AI

# 07/10/2023 - Jakob De Vreese


# Code for extracting notes from MIDI files and pass it to the model trainer

# Import the necessary packages
import os
import mido
import torch


# Define a function to extract notes from midi files and return the neassesary information
def extract_notes():

    # Give feedback
    print("Data Preperation for Bach-AI")
    print("Starting...")
    print("")
    print("Extracting notes from MIDI files...")
    print("This may take a while...")
    print("")
    
    # File paths
    cwd = os.getcwd() # Get the current working directory

    fugue_folder = os.path.join(cwd, r"data\Fugas") # point to the right folder 
    theme_folder = os.path.join(cwd, r"data\Themas")

    # Give feedback
    print("Fugue folder: ", fugue_folder)
    print("Theme folder: ", theme_folder)
    print("")


    # Function extracting notes from a MIDI file
    def extract_notes_from_track(midi_file, track_number, theme_track_number):
    
        notes = []  # list of tensors containing the pitch, length, and track number of each note
        midi = mido.MidiFile(midi_file)  # load the MIDI file

        # Maintain a dictionary to keep track of active notes and their start times
        active_notes = {}

        for msg in midi.tracks[track_number]:
            if msg.type == 'note_on':
                if track_number == 0:  # if it is a theme track
                    track_num = theme_track_number	
                else:
                    track_num = track_number

                # Check if the message contains the correct information
                if isinstance(msg.note, int) and isinstance(msg.time, int):
                    pitch = msg.note
                    start_time = msg.time

                    # Check if the note is already active
                    if pitch in active_notes:
                        # calculate the duration of the note
                        duration = start_time - active_notes[pitch]
                        note = torch.tensor([pitch, duration, track_num], dtype=torch.int32)
                        notes.append(note)
                
                    else:
                        # Store the start time of the active note
                        active_notes[pitch] = start_time
        
        return notes


    # Initialize the lists to store the notes
    fugue_list = []
    theme_list = []

    # Initialize variable for counting the number of fugues
    fugue_count = 0

    # Iterate through the Midi files in the fugue folder
    for fugue_filename in os.listdir(fugue_folder):
        fugue_count += 1
        fugue_full_path = os.path.join(fugue_folder, fugue_filename)
        midi = mido.MidiFile(fugue_full_path)

        # Iterate through the tracks in the MIDI file
        for track_number, track in enumerate(midi.tracks):
            track_notes = extract_notes_from_track(fugue_full_path, track_number, track_number) # when it is a fugue and there is a track on 0 then it is certain the track number stays correct
        
            # Check if the track contains notes
            if len(track_notes) > 0:
                # Find the corresponding theme file
                theme_filename = fugue_filename.replace('.mid', '_theme.mid')
                theme_full_path = os.path.join(theme_folder, theme_filename)

                # Check if the theme file exists before trying to extract notes from it
                if os.path.exists(theme_full_path):
                    theme_notes = extract_notes_from_track(theme_full_path, 0, track_number) # to give the theme track number the track_number of the fugue track

                    # Filter out unwanted notes - Notes with pitch 0 or negative pitch, notes with negative duration
                    theme_notes = [note for note in theme_notes if len(note) > 0 and 1 <= note[0] <= 127 and note[1] > 0] 
                    track_notes = [note for note in track_notes if len(note) > 0 and 1 <= note[0] <= 127 and note[1] > 0]
                
                    fugue_list.append(track_notes)
                    theme_list.append(theme_notes)

                    # Print the progress
                    print("Fugue and theme added to lists")
                    print("name:" + fugue_filename + " track:" + str(track_number))
                    print("theme:" + theme_filename + " " + str(track_number))              

    # Give feedback
    print("")
    print("Data processing finnished")
    print("")
    print("number of fugues: " + str(fugue_count))
    print("")
    print("number of rows fugue list: " + str(len(fugue_list)))
    print("number of rows theme list: " + str(len(theme_list)))
    print("")
    print("")
    print("")
    print("Data Preperation for Bach-AI done!")
    print("Step 1 completed! What a relief!")
    print("")

    return fugue_list, theme_list