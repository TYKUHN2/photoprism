package facerec

import (
"encoding/json"
"errors"
"github.com/Kagami/go-face"
"image"
"log"
"os"
"sync"
"sync/atomic"
)

type savedFace struct {
	name string
	descriptor face.Descriptor
}

type UnknownFace struct {
	Rect image.Rectangle
}

type KnownFace struct {
	Name string
	Rect image.Rectangle
}

type EZRecognizer struct {
	refs uint32
	mutex sync.RWMutex

	rec *face.Recognizer
	labels []string
}

var instance EZRecognizer

func main() {
	rec, err := CreateRecognizer()
	if err != nil {
		log.Fatalf("Unable to create recognizer: %v", err)
	}
	defer rec.Close()

	known, unknown, err := rec.Recognize("test.jpg")
	if err != nil {
		log.Fatalf("Unable to run recognizer: %v", err)
	}

	var names []string
	for _, v := range known {
		if find(names, v.Name) == -1 {
			names = append(names, v.Name)
		}
	}

	log.Printf("I saw %v as well as %v unknown faces", names, len(unknown))
}

func CreateRecognizer() (*EZRecognizer, error) {
	newRefs := atomic.AddUint32(&instance.refs, 1)
	if newRefs == 1 && instance.rec == nil {
		instance.mutex.Lock()
		defer instance.mutex.Unlock()

		var err error
		instance.rec, err = face.NewRecognizer("models")
		if err != nil {
			return nil, err
		}

		var samples []face.Descriptor
		var cats []int32
		samples, cats, instance.labels, err = decodeSave()
		if err != nil {
			return nil, err
		}

		instance.rec.SetSamples(samples, cats)
	}

	return &instance, nil
}

func (*EZRecognizer) Close() {
	newRefs := atomic.AddUint32(&instance.refs, ^0)

	if newRefs == 0 && instance.rec != nil {
		instance.mutex.Lock()

		instance.rec.Close()
		instance.rec = nil

		instance.mutex.Unlock()
	}
}

func (*EZRecognizer) Train(img string, name string) error {
	instance.mutex.Lock()
	defer instance.mutex.Unlock()

	faces, err := instance.rec.RecognizeFile(img)
	if err != nil {
		return err
	}

	var unknownFace *face.Face
	for _, f := range faces {
		id := instance.rec.Classify(f.Descriptor)

		if id < 0 {
			if unknownFace != nil {
				return errors.New("more than one unknown face in image")
			}
			unknownFace = &f
		}
	}

	if unknownFace == nil {
		return errors.New("image has no unknown faces")
	}

	var samples []face.Descriptor
	var cats []int32
	samples, cats, instance.labels, err = decodeSave()
	if err != nil {
		return err
	}

	samples = append(samples, unknownFace.Descriptor)

	index := find(instance.labels, name)
	if index == -1 {
		index = int32(len(instance.labels))
		instance.labels = append(instance.labels, name)
	}

	cats = append(cats, index)

	instance.rec.SetSamples(samples, cats)

	return encodeSave(samples, cats, instance.labels)
}

func (*EZRecognizer) TrainFace(img string, trainFace UnknownFace, name string) error {
	instance.mutex.Lock()
	defer instance.mutex.Unlock()

	faces, err := instance.rec.RecognizeFile(img)
	if err != nil {
		return err
	}

	var unknownFace *face.Face
	for _, f := range faces {

		if f.Rectangle == trainFace.Rect {
			unknownFace = &f
			break
		}
	}

	if unknownFace == nil {
		return errors.New("face not found in image")
	}

	var samples []face.Descriptor
	var cats []int32
	samples, cats, instance.labels, err = decodeSave()
	if err != nil {
		return err
	}

	samples = append(samples, unknownFace.Descriptor)

	index := find(instance.labels, name)
	if index == -1 {
		index = int32(len(instance.labels))
		instance.labels = append(instance.labels, name)
	}

	cats = append(cats, index)

	instance.rec.SetSamples(samples, cats)

	return encodeSave(samples, cats, instance.labels)
}

func (*EZRecognizer) Recognize(img string) ([]KnownFace, []UnknownFace, error) {
	instance.mutex.RLock()
	defer instance.mutex.RUnlock()

	faces, err := instance.rec.RecognizeFile(img)
	if err != nil {
		return nil, nil, err
	}

	var knownFaces []KnownFace
	var unknownFaces []UnknownFace
	for _, f := range faces {
		id := instance.rec.Classify(f.Descriptor)

		if id >= 0 {
			knownFaces = append(knownFaces, KnownFace{instance.labels[id], f.Rectangle})
		} else {
			unknownFaces = append(unknownFaces, UnknownFace{ f.Rectangle })
		}
	}

	return knownFaces, unknownFaces, nil
}

func encodeSave(samples []face.Descriptor, cats []int32, labels []string) error {
	if len(samples) != len(cats) {
		return errors.New("mismatched samples and categories length")
	}

	nLabels := int32(len(labels))
	for _, cat := range cats {
		if cat >= nLabels {
			return errors.New("category without attached label")
		}
	}

	var savedFaces []savedFace
	for i := 0; i < len(samples); i++ {
		savedFaces = append(savedFaces, savedFace{ labels[cats[i]], samples[i]})
	}

	return writeSave(savedFaces)
}

func writeSave(faces []savedFace) error {
	file, err := os.Open("known.json")
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)

	for _, tFace := range faces {
		err := encoder.Encode(&tFace)
		if err != nil {
			return err
		}
	}

	return nil
}

func decodeSave() ([]face.Descriptor, []int32, []string, error) {
	savedFaces, err := readSave()
	if err != nil {
		return nil, nil, nil, err
	}

	var samples [] face.Descriptor
	var cats []int32
	var labels []string
	for _, f := range savedFaces {
		index := find(labels, f.name)
		if index == -1 {
			index = int32(len(labels))
			labels = append(labels, f.name)
		}

		samples = append(samples, f.descriptor)
		cats = append(cats, index)
	}

	return samples, cats, labels, nil
}

func readSave() ([]savedFace, error) {
	file, err := os.Open("known.json")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()

	var savedFaces []savedFace
	for decoder.More() {
		var tFace savedFace

		err := decoder.Decode(&tFace)
		if err != nil {
			return nil, err
		}

		savedFaces = append(savedFaces, tFace)
	}

	return savedFaces, nil
}

func find(arr []string, str string) int32 {
	for i, s := range arr {
		if s == str {
			return int32(i)
		}
	}
	return -1
}
