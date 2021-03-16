package facerec

import (
	"encoding/json"
	"errors"
	"github.com/Kagami/go-face"
	"github.com/photoprism/photoprism/internal/config"
	"image"
	"os"
	"path/filepath"
)

type savedFace struct {
	name       string
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
	rec    *face.Recognizer
	cfg    *config.Config
	labels []string
}

func CreateRecognizer(cfg *config.Config) (*EZRecognizer, error) {
	instance := EZRecognizer{}
	instance.cfg = cfg

	rec, err := face.NewRecognizer("internal/facerec/models")
	if err != nil {
		return nil, err
	}

	samples, cats, labels, err := decodeSave(cfg)
	if err != nil {
		rec.Close()
		return nil, err
	}

	rec.SetSamples(samples, cats)

	return &EZRecognizer{rec, cfg, labels}, nil
}

func (instance EZRecognizer) Close() {
	instance.rec.Close()
}

func (instance EZRecognizer) Train(img string, name string) error {
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
	samples, cats, instance.labels, err = decodeSave(instance.cfg)
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

	return encodeSave(samples, cats, instance.labels, instance.cfg)
}

func (instance EZRecognizer) TrainFace(img string, trainFace UnknownFace, name string) error {
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
	samples, cats, instance.labels, err = decodeSave(instance.cfg)
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

	return encodeSave(samples, cats, instance.labels, instance.cfg)
}

func (instance EZRecognizer) Recognize(img string) ([]KnownFace, []UnknownFace, error) {
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
			unknownFaces = append(unknownFaces, UnknownFace{f.Rectangle})
		}
	}

	return knownFaces, unknownFaces, nil
}

func encodeSave(samples []face.Descriptor, cats []int32, labels []string, cfg *config.Config) error {
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
		savedFaces = append(savedFaces, savedFace{labels[cats[i]], samples[i]})
	}

	return writeSave(savedFaces, cfg)
}

func writeSave(faces []savedFace, cfg *config.Config) error {
	file, err := os.Create(getConfig(cfg))
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

func decodeSave(cfg *config.Config) ([]face.Descriptor, []int32, []string, error) {
	savedFaces, err := readSave(cfg)
	if err != nil {
		return nil, nil, nil, err
	}

	var samples []face.Descriptor
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

func readSave(cfg *config.Config) ([]savedFace, error) {
	file, err := os.Open(getConfig(cfg))
	if err != nil {
		if os.IsNotExist(err) { // Suppress non-exist because we can just treat it as empty
			return nil, nil
		} else {
			return nil, err
		}
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

func getConfig(cfg *config.Config) string {
	return filepath.Join(cfg.StoragePath(), "facerec.json")
}
