# whatsinmypics

whatsinmy.pics recommends tags for images uploaded to a service like Flikr or Instagram. It has two main components

* A pre-trained neural network (http://places.csail.mit.edu) which identifies scenery
* A topic model trained on 1.5M images of the Flickr 100M photo dataset

A user uploads a photo, is given a set of suggested tags that they can approve, reject, or augment, then can search
for additional relevant photos based on the photo and provided tags.
