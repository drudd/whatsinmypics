$(function() { Dropzone.options.drop = {
  paramName: "file", 
  maxFilesize: 4, // MB
  uploadMultiple: false,
  addRemoveLinks: false,
  acceptedFiles: ".jpg,.png",
  previewTemplate: "<div class=\"dz-preview dz-file-preview row\">\n <div class=\"dz-image\"><img data-dz-thumbnail class=\"dz-image center-block\"/></div>\n </div> </div>\">",
  init: function() {
    this.on("success", function(file, responseText) {
      alert(responseText);
    });
	this.on("addedfile", function(file) { 
	  $(".dz-message").detach(); $(".dz-preview").slice(0,-1).remove();
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
