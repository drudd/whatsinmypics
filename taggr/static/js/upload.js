$(function() { Dropzone.options.drop = {
  paramName: "file", 
  maxFilesize: 4, // MB
  uploadMultiple: false,
  addRemoveLinks: false,
  acceptedFiles: ".jpg,.png",
  previewTemplate: "<div class=\"dz-preview dz-file-preview row\">\n <div class=\"dz-image\"><img data-dz-thumbnail class=\"dz-image center-block\"/></div>\n </div> </div>\">",
  init: function() {
    this.on("success", function(file, responseText) {
	  var response = $.parseJSON(responseText);
	  //document.createElement("<div class=\"row\"> 
	  //alert(response.suggested_tags);
      var tag_html = "<div class=\"col-md-8 centered text-center\">"; //<div class=\"btn-group btn-group-justified\">";
	  var tags = response.suggested_tags;
	  for (var i = 0; i < tags.length; i++) {
		tag_html += "<button type=\"button\" class=\"btn btn-default\" data-toggle=\"button\">"+tags[i]+"</button>";
	  }
	  tag_html += "</div>" //</div>";
      tag_html += "<div class=\"col-md-8 centered text-center\">"; //<div class=\"btn-group btn-group-justified\">";
      var images = response.suggested_images;
      for (var i = 0; i < images.length; i++) {
        tag_html += "<img src=\""+images[i]+"\">";
      }
      tag_html += "</div>" //</div>";

	  tagElement = document.createElement("div");
	  tagElement.classList.add("row");
	  tagElement.innerHTML = tag_html;
	  document.body.appendChild(tagElement);
    });
	this.on("addedfile", function(file) { 
	  $(".dz-message").detach(); $(".dz-preview").slice(0,-1).remove();
	  $(".results").slice(0,-1).remove();
	});
  },
  resize: function(file) {
	var scale = 250./Math.max(file.width, file.height)
    var resizeInfo = {
          srcX: 0,
          srcY: 0,
          trgX: 0,
          trgY: 0,
          srcWidth: file.width,
          srcHeight: file.height,
          trgWidth: scale*file.width,
          trgHeight: scale*file.height 
    };

    return resizeInfo;
  }
}});
$(document).bind("dragover", function(e) {
            e.preventDefault();
            return false;
       });

$(document).bind("drop", function(e){
            e.preventDefault();
            return false;
        });
