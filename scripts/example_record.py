from ble2lsl.devices import muse2016
import ble2lsl as b2l

from wizardhat import acquired

# using a Dummy instead of a
streamer = b2l.Dummy(muse2016)

# this will start writing data for all streams ('EEG', 'telemetry', etc.) to disk
# "label" will be added to any filenames associated with this Receiver
receiver = acquire.Receiver(label='session_1')

# in a real experiment, there may be some setup at this point;
# e.g. to control when to record a given duration

# this will create a new CSV/JSON file pair; the data for the next 5 seconds
# will continue to be saved to the primary file created when the
# Receiver was created, but will also write this interval of data
# to the new file, which will be labelled with the Receiver label ('session_1'
# in this case) as well as the recording label ('first_trial')
while True:
    trial_label = input("Enter a trial label to start a recording.\n" +
                        "Enter 'stop' to exit program")
    if trial_label == 'stop':
        break
    else:
        duration = input("Enter a recording duration (default=5 sec). " +
                         "The recording will begin when you press Enter: ")
        if duration == '':
            # use the default if the user didn't choose
            duration = 5
        else:
            duration = int(duration)
    receiver.record(duration=duration, label=trial_label)
