$(function() { Dropzone.options.drop = {
  paramName: "file", 
  maxFilesize: 4, // MB
  uploadMultiple: false,
  addRemoveLinks: false,
  acceptedFiles: ".jpg,.png",
  previewTemplate: "<div class=\"dz-preview dz-file-preview\">\n <div class=\"dz-progress\"></div> <img data-dz-thumbnail class=\"dz-image\"/></div>\n </div>\">",
  init: function() {
    this.on("success", function(file, responseText) {
	  $('.dz-progress').circleProgress('value', 1.0);
	  var response = $.parseJSON(responseText);
	  process_image(response);	
	  $(".dz-progress").remove();
    });
	this.on("addedfile", function(file) { 
		added_file();
	});
	this.on("uploadprogress", function(file,progress,bytesSent) {
		if ( progress > 5 ) {
			$('.dz-progress').circleProgress('value', Math.min(0.95, progress/100.));
		}
	});
	this.on("sending", function(file) {
	    $(".dz-progress").circleProgress({
			value: 0.05,
			size: 150,
			fill: {
				gradient: ["blue", "white"]
			}
		}).on('circle-animation-progress', function(event, progress) {
			if ( progress > 0.5 ) {
				$(this).find('strong').html('<i>Classifying</i>');
			}
		});
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

$(".random-button").click(function(e) {
	$.getJSON("/random", function( data ) {
      var img;
      img = document.createElement("img");
	  img.classList.add("dz-image");
      img.src = data.input_file;
	  img.onload = (function(this_) {
			var scale = 250./Math.max(this_.width, this_.height)
			this_.width = scale*this_.width;
			this_.height = scale*this_.height;
		});

	  child = document.createElement("div");
	  child.classList.add("dz-preview");
	  child.appendChild(img);
	  document.getElementById("drop").appendChild(child);

	  added_file();	
	  process_image(data);
    });
	
	e.stopPropagation();
});

function process_image(response) {
    var tag_html = "<div class=\"container-fluid tagform\"><select class=\"tag-select form-control\" multiple=\"multiple\">"; 
	var tags = response.suggested_tags;
	for (var i = 0; i < tags.length; i++) {
		tag_html += "<option selected>"+tags[i]+"</option>";
	}
	tag_html += "</select>\n";
	//tag_html += "<button type=\"button\" class=\"btn 
	tag_html += "</div><br/>";

	var images = response.suggested_images;
	tag_html += "<div class=\"container-fluid carousel-container\"><div id=\"carousel\" class=\"carousel slide\" data-ride=\"carousel\"><ol class=\"carousel-indicators\">\n";
	for (var i = 0; i < images.length; i++) {
		tag_html += "<li data-target=\"#carousel\" data-slide-to=\"" + i + "\"></li>\n";
	}
	tag_html += "</ol><div class=\"carousel-inner\" role=\"listbox\">\n";
	for (var i = 0; i < images.length; i++) {
        tag_html += "<div class=\"item" + ((i==0) ? " active" : "") + "\"><img src=\""+images[i]+"\"></div>\n";
    }
	
	tag_html += "<a class=\"left carousel-control\" href=\"#corousel\" role=\"button\" data-slide=\"prev\">\n";
    tag_html += "  <span class=\"glyphicon glyphicon-chevron-left\" aria-hidden=\"true\"></span>\n";
	tag_html += "  <span class=\"sr-only\">Previous</span>\n";
    tag_html += "</a>\n";
    tag_html += "<a class=\"right carousel-control\" href=\"#carousel\" role=\"button\" data-slide=\"next\">\n";
    tag_html += "  <span class=\"glyphicon glyphicon-chevron-right\" aria-hidden=\"true\"></span>\n";
    tag_html += "  <span class=\"sr-only\">Next</span>\n";
    tag_html += "</a>\n";
	tag_html += "</div></div>\n";

	tagElement = document.createElement("div");
	tagElement.classList.add("row");
	tagElement.classList.add("results");
	tagElement.innerHTML = tag_html;
	document.body.appendChild(tagElement);

	$(".tag-select").select2({
	  tags: true,
	  tokenSeparators: [',']
	});
	
	$(".dz-progress").remove();
}

function added_file() {
	$(".dz-message").detach();
	$(".dz-preview").slice(0,-1).remove();
	$(".results").remove();
}

$(document).bind("dragover", function(e) {
            e.preventDefault();
            return false;
       });

$(document).bind("drop", function(e){
            e.preventDefault();
            return false;
        });

