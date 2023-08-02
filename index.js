var xhttp = new XMLHttpRequest();
xhttp.onreadystatechange =  function() {
    if (this.readyState == 4 && this.status == 200) {
        var xmlResp = this.responseXML;

        var header_markup = xmlResp.getElementsByTagName("headerHtml")[0];
        var footer_markup = xmlResp.getElementsByTagName("footerHtml")[0];


        if (document.getElementById("UHF-header") != null){
            document.getElementById("UHF-header").innerHTML += header_markup.textContent;
        }
        else {
            document.getElementById("UHF-header-about").innerHTML += header_markup.textContent;
        }

        if (document.getElementById("UHF-footer") != null){
            document.getElementById("UHF-footer").innerHTML += footer_markup.textContent;
        }
        else{
            document.getElementById("UHF-footer-about").innerHTML += footer_markup.textContent;
        }

        var css_markup = xmlResp.getElementsByTagName("cssIncludes")[0];

        var head = document.getElementsByTagName("head")[0];
        head.innerHTML += css_markup.textContent;
        console.log(header_markup.textContent);
    }
};

$(document).ready(function() {
  // Set the default tab as active
  $(".tab-content").hide();
  $("#tab1").show();
  $(".tab-button[data-tab='tab1']").addClass("active focus");

  // Add event listeners to all tab buttons
  $(".tab-button").click(function() {
    var tabName = $(this).attr("data-tab");

    // Hide all tab content
    $(".tab-content").hide();

    // Remove "active" class from all tab buttons
    $(".tab-button").removeClass("active");

    // Show the clicked tab content and set the button as active
    $("#" + tabName).show();
    $(this).addClass("active focus");
  });
});
   
xhttp.open("GET", "https://uhf.microsoft.com/en-US/shell/xml/MSDirectML?headerId=MSDirectMLHeader&footerid=MSDirectMLFooter&CookieComplianceEnabled=true", true);
xhttp.send();
