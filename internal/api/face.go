package api

import (
	"github.com/gin-gonic/gin"
	"github.com/photoprism/photoprism/internal/acl"
	"github.com/photoprism/photoprism/internal/facerec"
	"github.com/photoprism/photoprism/internal/i18n"
	"github.com/photoprism/photoprism/internal/photoprism"
	"github.com/photoprism/photoprism/internal/query"
	"github.com/photoprism/photoprism/internal/service"
	"github.com/photoprism/photoprism/pkg/fs"
	"github.com/photoprism/photoprism/pkg/txt"
	"image"
	"net/http"
	"strconv"
)

func TrainFace(router *gin.RouterGroup) {
	router.POST("/trainface/:uid", func(c *gin.Context) {
		conf := service.Config()

		if conf.ReadOnly() || conf.DisableFaceRecognition() {
			Abort(c, http.StatusForbidden, i18n.ErrReadOnly)
		}

		s := Auth(SessionID(c), acl.ResourcePhotos, acl.ActionSearch)

		if s.Invalid() {
			AbortUnauthorized(c)
			return
		}

		f, err := query.FileByPhotoUID(c.Param("uid"))

		if err != nil {
			c.Data(http.StatusNotFound, "image/svg+xml", photoIconSvg)
			return
		}

		fileName := photoprism.FileName(f.FileRoot, f.FileName)

		if !fs.FileExists(fileName) {
			log.Errorf("photo: file %s is missing", txt.Quote(f.FileName))
			c.Data(http.StatusNotFound, "image/svg+xml", photoIconSvg)

			// Set missing flag so that the file doesn't show up in search results anymore.
			logError("photo", f.Update("FileMissing", true))

			return
		}

		rec, err := facerec.CreateRecognizer(conf)
		if err != nil {
			AbortUnexpected(c)
			return
		}

		known, unknown, err := rec.Recognize(fileName)
		if err != nil {
			AbortUnexpected(c)
			return
		}

		form, err := c.MultipartForm()

		if err != nil {
			AbortBadRequest(c)
			return
		}

		min, err := readPoint(form.Value["min"])
		if err != nil {
			AbortBadRequest(c)
			return
		}

		max, err := readPoint(form.Value["max"])
		if err != nil {
			AbortBadRequest(c)
			return
		}

		sel := image.Rectangle{ Min: min, Max: max }
		var selFace *facerec.UnknownFace
		for _, face := range unknown {
			if face.Rect.In(sel) || sel.In(face.Rect) {
				if selFace != nil {
					c.Data(http.StatusNotFound, "text/plain", []byte("multiple faces"))

					// Set missing flag so that the file doesn't show up in search results anymore.
					logError("photo", f.Update("FileMissing", true))

					return
				}
				selFace = &face
			}
		}

		if selFace == nil {
			for _, face := range known {
				if face.Rect.In(sel) || sel.In(face.Rect) {
					AbortAlreadyExists(c, "face known")
					return
				}
			}

			c.Data(http.StatusNotFound, "text/plain", []byte("no faces"))

			// Set missing flag so that the file doesn't show up in search results anymore.
			logError("photo", f.Update("FileMissing", true))

			return
		}

		name := form.Value["name"][0]

		err = rec.TrainFace(fileName, *selFace, name)
		if err != nil {
			AbortUnexpected(c)
			return
		}
	})
}

func readPoint(str []string) (image.Point, error) {
	x, err := strconv.Atoi(str[0])
	if err != nil {
		return image.Point{}, err
	}

	y, err := strconv.Atoi(str[0])
	if err != nil {
		return image.Point{}, err
	}

	return image.Point{ X: x, Y: y }, nil
}
