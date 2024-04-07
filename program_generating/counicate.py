import mido
import time
mido.set_backend('mido.backends.portmidi')

def play_note(port, note, duration=1):
    velocity = 100
    ## Message MIDI
    msg_on = mido.Message('note_on', note = note, velocity = velocity)
    port.send(msg_on)
    time.sleep(1)
    
    ## Message MIDI
    msg_off = mido.Message('note_off', note= note)
    port.send(msg_off)

def receive_note(port):
    msg = port.receive()
    print(msg.note)
    if "note_on" is msg.type : 
        return msg.note
    return 200

def main():
    
    ports = mido.get_output_names()
    print("Ports MIDI disponibles:", ports)
    port_name = [name for name in ports if 'Disklavier' in name]
    notes = []
    if not port_name:
        print("Piano Disklavier non trouvé.")
        return
    with mido.open_input(port_name[0]) as port:
        print(f"Enregristrement des notes{port_name[0]}")
        i = 0 
        while i <10:
            cle = receive_note(port)
            if cle != 200: 
                notes.append(cle)
                i= i+1
            

    print(notes)
    with mido.open_output(port_name[0]) as port:
        print(f"Envoi de notes au port {port_name[0]}")
        for cle in notes: play_note(port, cle, duration=0.5)
        

if __name__ == "__main__":
    main()