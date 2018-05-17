import pretty_midi

def get_avg_velocity(instrument):
    # 平均力度
    velocity = 0
    for note in instrument.notes:
        velocity += note.velocity
    velocity /= len(instrument.notes)
    return velocity

def get_sum_duration(instrument):
    # 所有音符累加总时值
    duration = 0
    for note in instrument.notes:
        duration += note.end - note.start
    return duration

def get_types_of_duration(instrument):
    # 时值类型
    duration = []
    for note in instrument.notes:
        duration.append(note.end - note.start)
    types = len(set(duration))
    return types

def get_pitch_range(instrument):
    # 最高音与最低音之间的音程
    pitch = []
    for note in instrument.notes:
        pitch.append( note.pitch )
    sorted_pitch = sorted( set(pitch) )

    range = sorted_pitch[-1] - sorted_pitch[0]
    return range

def get_second_pitch_range(instrument):
    # 第二高音与第二低音之间的音程
    pitch = []
    for note in instrument.notes:
        pitch.append( note.pitch )
    sorted_pitch = sorted( set(pitch) )

    range = sorted_pitch[-2] - sorted_pitch[1]
    return range

def get_train_example(midi_file):
    pm = pretty_midi.PrettyMIDI( midi_file=midi_file )

    train_example = []

    for instrument in pm.instruments:
        train_example.append(get_avg_velocity(instrument))
        train_example.append(get_sum_duration(instrument))
        train_example.append(get_types_of_duration(instrument))
        train_example.append(get_pitch_range(instrument))
        train_example.append(get_second_pitch_range(instrument))

    return train_example

def get_X_train():
    X_train = [[]]
    for i in range(13): # 一共切取了14个midi片段，其中13个作为训练样本
        midi_file = 'data/x'+str(i+1)+'.mid'
        train_example = get_train_example(midi_file)
        if i==0:
            X_train[0]=train_example
        else:
            X_train.append(train_example)
    return X_train

def get_X_test():
    X_test = [[]]

    midi_file = 'data/x14.mid'
    train_example = get_train_example(midi_file)
    X_test[0]=train_example
    return X_test


