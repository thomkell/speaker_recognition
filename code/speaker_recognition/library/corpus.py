import os
import re
from collections import defaultdict

class King:
    """
    King corpus manipulator
    Can be used to access files associcated with a specific speaker

    Sample usage:
    king = King("path/to/files")
    print(f"Speakers:  {king.get_speakers()}")

    Assuming speaker 7 is one of the speakers:
    speaker = 7
    print(f"Files associated with speaker {speaker}={king[speaker]}")
    """
    # Regexp to match King files and decode.
    # Expects:
    # w|n - wide or narrow-band marker
    # digits for session number _
    # digits for speaker number _
    # digits for conversation topic
    utterance_re = re.compile(
        r"(?P<bandwidth>[wn])(?P<session>\d+)_(?P<speaker>\d+)_(?P<topic>\d).*")

    def __init__(self, directory:str, min_utterances=10, group="Nutley") -> None:
        """
        Initialize a King corpus object
        :param directory:  Directory containing audio
        :param min_utterances: Speakers that have < min_utterances files
          will be discarded.
        :param group: speaker group
           "all" - all speakers
           "San Diego" - Speakers 1-26 from San Diego, CA ITT facility
           "Nutley" - "Speakers 27-51 from Nutley, NJ ITT facility
           Any speaker within a group that does not meet the min_utterances
           criteria will be discarded.
        """

        # Set up an anonymous function to test group membership
        # This allows us to include/reject speakers based on the group
        if group == "all":
            speaker_test = lambda s : True
        elif group == "San Diego":
            speaker_test = lambda s : s <= 26
        elif group == "Nutley":
            speaker_test = lambda s: s >= 27
        else:
            raise ValueError(f"Bad group value: {group}")

        # Dictionary with filenames for each speaker
        # defaults to a new list the first time a speaker is accessed
        self.speaker_data = defaultdict(list)
        files_n = 0  # Number of files
        for path, directory, files in os.walk(directory):
            # Only keep files that meet criteria
            for f in files:
                # See if we meet the criteria to match this file
                match = self.utterance_re.match(f)
                if match:
                    # Add file to list for specified speaker
                    speaker = int(match.group("speaker"))
                    if speaker_test(speaker):
                        utterances = self.speaker_data[speaker]
                        utterances.append(os.path.join(path, f))
                        files_n = files_n + 1  # one more file processed

        # See if we need to discard any speakers who do not have enough files.
        speakers = list(self.speaker_data.keys())
        for s in speakers:
            # How many files for speaker s?
            utterances_n = len(self.speaker_data[s])
            if utterances_n < min_utterances:
                print(f"Discarding speaker {s} with {utterances_n}")
                del self.speaker_data[s]

        # Summary
        print(f"{len(self.speaker_data)} speakers with {files_n} files")

        # Some speaker numbers are skipped.  When building a classifier
        # we will want to renumber all speakers between 0 and N-1.
        # Construct a map to do this.
        speakers = list(self.speaker_data.keys())
        speakers.sort()  # Ensure in order to make things easier
        self.speaker_2_category = {}
        self.category_2_speaker = {}
        category = 0
        for s in speakers:
            self.speaker_2_category[s] = category
            self.category_2_speaker[category] = s
            category = category + 1

        return

    def __getitem__(self, speaker) -> list[str]:
        """
        Return audio files for specified speaker, e.g. king[23]
        :param speaker: speaker number
        :return: list of files
        """

        if speaker in self.speaker_data:
            filelist = self.speaker_data[speaker]
        else:
            raise ValueError(f"Speaker {speaker} does not exist")
        return filelist


    def get_speakers(self) -> list[int]:
        """
        Return a list of speakers for which we have data
        :return:
        """

        return self.speaker_data.keys()

    def speaker_category(self, s: int) -> int:
        """
        Category 0 to N-1 assigned to speaker s
        :param s: speaker identifier
        :return: category
        """
        return self.speaker_2_category[s]

    def category_to_speaker(self, c: int) -> int:
        """
        Return the speaker associated with category c
        :param c: category index
        :return: speaker label
        """
        return self.category_2_speaker[c]

    def get_categories_N(self):
        """
        :return: Number of categories in corpus
        """
        # Number of speakers
        return len(self.speaker_data)
