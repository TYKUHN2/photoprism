package facerec

import (
	"os"
	"testing"
)

func TestImageRecognition(t *testing.T) {
	rec, err := CreateRecognizer()
	if err != nil {
		t.Fatalf("Failed to create recognizer: %v", err)
	}
	defer rec.Close()

	known, unknown, err := rec.Recognize("test.jpg")
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("test image was not available, skipping")
		} else {
			t.Fatalf("failed to recognize test image: %v", err)
		}
	}

	var names []string
	for _, v := range known {
		if find(names, v.Name) == -1 {
			names = append(names, v.Name)
		}
	}

	t.Logf("found %v and %v unknown faces", names, len(unknown))
}
