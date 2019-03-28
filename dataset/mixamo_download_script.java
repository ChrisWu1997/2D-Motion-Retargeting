/*
INSTRUCTIONS:
1. Set your browser NOT to ask where to save files on each download (else you will be bombarded with "Save As" prompts!)
2. Navigate to https://www.mixamo.com/#/?page=1&type=Motion%2CMotionPack&limit=96
3. For best results, set on-page  thumbnails to "small" and "non-animated" (click the little gear icon)
4. Set to display 96 items per page (you can also enter 100 in the url, but it makes not much of a difference)
5. IMPORTANT: Before you run the script, make sure to manually download one sample animation - the page will remember your settings (if you don't do this, it will use defaults, i.e. it will download the character with each anim)
6. Set one (and only 1) of the first 6 variables to TRUE, in order to select the variant you want to download (regular, in place, root) by default it will download only the regular version (without inplace option).
7. Open the browser console (Shift+Ctrl+J in Chrome) and paste the entirety of the code
8. Hit Enter and wait for it to finish (can take some time)

NOTE: the script skips all of mixamos animation "packs", as those are duplicates (their content is available as seperate animations), but mostly because it messed up point no. 5

It is RECOMMENDED to use the variables below in the following WORKFLOW:
(1) set the first variable (Setup_DownloadAllExceptInPlace) from false to true,
(2) run the script, (3) move all downloaded animations to folder named "\regular",
(4) set the first variable back to false and the second to true,
(5) run the script again (make sure to open in a fresh tab and don't forget to repeat 5.),
(6) move all downloaded animations to folder named "\inplace",
(7) repeat for 3rd variable to have the entire set, or all 6 to have also a mirrored version.

You should end up with 2061 regular animations and 384 in root and inplace each.

TROUBLESHOOTING
With slower connections it is possible that your download queue will start to choke and leave downloads unfinished.
If that happens, reduce the per page amount of animations from 96 to 24, or manually enter 10 in the URL (like this) - on each page turn the script waits for a couple of seconds to help with the above.

For offline browsing purposes, here's a download with all thumbnail:
Here are the Thumbnails:
MixamoThumbnails.7z (523.28 MB)
(I've been made aware that the host uses pesky popup ads and such, keep that in mind or use an adblocker or better yet, a download manager, I highly recommend JDownloader2).

Have fun!
*/

/* these variables are there to allow running the script in multiple passes so the animations can then be grouped manually into organized folders */
var Setup_DownloadAllExceptInPlace                = true;    // example folder:         \regular
var Setup_DownloadOnlyInPlace                    = false;    // example folder:         \inplace
var Setup_DownloadOnlyRoot                        = false;    // example folder:         \root

var Setup_DownloadOnlyInPlace_Mirrored            = false;    // example folder:         \mirrored\inplace
var Setup_DownloadAllExceptInPlace_Mirrored        = false;    // example folder:         \mirrored\regular
var Setup_DownloadOnlyRoot_Mirrored                = false;    // example folder:         \mirrored\root


/* Debug Settings */

var debugVerbose                        = true;      // write messages to console
var debugMaxIndex                         = 0;        // will stop after x anims, 0 = unlimited
var debugStopAfterPage                = 999;
var debugNoPageTurn                   = false;     // will finish on current page    
var setupGlobalTimeOffset              = 0;           // all timers will be increased by x (increase if script has to retry often)
var setupTimeModifier                  = 2;          // all timers will multiply by x (increase if script has to retry often)
var setupDefaultDelay                  = 500;         // default time step when adding anims to assets (increase when encountering problems)
var setupPlaySounds                 = true;      // play sound notifications
var setupBeepOnAdd                  = false;     // notification when an animation has been added to assets
var setupBeepOnPage                 = true;      // notification when changing page
var TimeToReloadPage                 = 10000;

/* Do NOT change values below unless you know what you are doing - internal vars and setup constants*/

var    pageURL                         = "https://www.mixamo.com/#/?type=Motion%2CMotionPack"; // base url without page and limit data
var setupElementID_Products         = "product";
var setupElementID_Download          = "btn-block btn btn-primary";
var setupElementID_DownloadConfirm     = "btn btn-primary";
var setupElementID_InPlace          = "inplace";
var setupElementID_Mirror           = "mirror";
var setupElementID_Products         = "product";
var setupElementID_Pagination       = ".pagination li:nth-last-child(2) a";
var setupElementID_DownloadCaption = "Download";
var DownloadPending_Regular         = downloadRegularVariant;
var DownloadPending_Mirror            = downloadMirrorVariant;
var DownloadPending_InPlace            = downloadInPlaceVariant;
var DownloadPending_InPlaceMirror    = downloadInPlaceMirrorVariant;
var timeEnd = 0, timeStart = 0;
var scriptPaused                      = true;
var index                           = -1;
var debugFailedStart                  = 0;
var pageNum                           = geUrlPageNum();
var itemsPerPage                       = geUrlLimit();
var NumOfFilesProcessed              = 0;
var buttonClicked                     = 0;
var retryCounter                     = 0;
var    button                          = document.querySelector( "setupElementID_AddToAssets" );
var items                           = document.getElementsByClassName( setupElementID_Products );

var    pageMax                                 = 5;        // limit pages
var downloadRegularVariant                     = true;     // no checkboxes set, regular version, i.e. root or default
var download_RootExclude                     = false;     // affects "downloadRegularVariant" only, ignores anims with "in place" option
var download_RootExclussive                 = false;     // affects "downloadRegularVariant" only, ignores anims without "in place" option
var download_RootMirrored                     = false;     // affects "downloadRegularVariant" only, forces mirror, meant to be used with "download_RootExclu[de|ssive]"
var downloadMirrorVariant                     = true;        // will check only "mirror" and download a separate version
var downloadInPlaceVariant                     = true;        // set In Place to true
var downloadInPlaceMirrorVariant             = true;        // In Place and mirrored to true

function DownloadAllExceptInPlace(){
    //A. downloadRegularVariant + download_RootExclude     // download all animations that have no INPLACE checkbox         \regular
    downloadRegularVariant                     = true;
    download_RootExclude                     = true;
    download_RootExclussive                 = false;
    download_RootMirrored                     = false;
    downloadMirrorVariant                     = false;
    downloadInPlaceVariant                     = true;
    downloadInPlaceMirrorVariant             = false;
}    

function DownloadAllExceptInPlace_Mirrored(){
    //B. A. + download_RootMirrored                        // downloads A., but mirrored                                     \regular_mirrored
    downloadRegularVariant                     = true;
    download_RootExclude                     = true;
    download_RootExclussive                 = false;
    download_RootMirrored                     = true;
    downloadMirrorVariant                     = false;
    downloadInPlaceVariant                     = false;
    downloadInPlaceMirrorVariant             = false;
}    

function DownloadOnlyInPlace(){
    //C. downloadInPlaceVariant                             // all anims w/ a INPLACE checkbox (set true)                     \inplace
    downloadRegularVariant                     = false;
    download_RootExclude                     = false;
    download_RootExclussive                 = false;
    download_RootMirrored                     = false;
    downloadMirrorVariant                     = false;
    downloadInPlaceVariant                     = true;
    downloadInPlaceMirrorVariant             = false;
}

function DownloadOnlyInPlace_Mirrored(){
    // D. downloadInPlaceMirrorVariant                        // same as C., but morrored                                        \inplace_mirrored
    downloadRegularVariant                     = false;
    download_RootExclude                     = false;
    download_RootExclussive                 = false;
    download_RootMirrored                     = false;
    downloadMirrorVariant                     = false;
    downloadInPlaceVariant                     = false;
    downloadInPlaceMirrorVariant             = true;
}

function DownloadOnlyRoot(){
    //E. downloadRegularVariant + download_RootExclussive // download only anims with INPLACE checkbox (left to false)    \root
    downloadRegularVariant                     = true;
    download_RootExclude                     = false;
    download_RootExclussive                 = true;
    download_RootMirrored                     = false;
    downloadMirrorVariant                     = false;
    downloadInPlaceVariant                     = false;
    downloadInPlaceMirrorVariant             = false;
}

function DownloadOnlyRoot_Mirrored(){
    //F. E. + download_RootMirrored   // E. mirrored                        \root_mirrored
    downloadRegularVariant                     = true;
    download_RootExclude                     = false;
    download_RootExclussive                 = true;
    download_RootMirrored                     = true;
    downloadMirrorVariant                     = false;
    downloadInPlaceVariant                     = false;
    downloadInPlaceMirrorVariant             = false;
}

function geUrlLimit(){
    var result = getUrlVars()["limit"];
    if (result && debugVerbose) console.log(result);
    else {
        if (debugVerbose) console.log("[DEBUG] geUrlLimit :: param LIMIT not found.");
        result = "96";
    }    
    return result;
}    

function isAnimPack(){
    var str = getAnimName();
    str = str.toLowerCase();
    var n = str.search(" pack");
    return ( n > -1 );
}    

var DownloadPending_Any = true;
function incIndex(){
    if (debugVerbose) console.log("[DEBUG] incIndex :: Start; index:",index);
    ++index;                
    ++NumOfFilesProcessed;
    buttonClicked = 0;

    DownloadPending_Regular         = downloadRegularVariant;
    DownloadPending_Mirror            = downloadMirrorVariant;
    DownloadPending_InPlace            = downloadInPlaceVariant;
    DownloadPending_InPlaceMirror    = downloadInPlaceMirrorVariant;
    DownloadPending_Any                = true;

    if (debugVerbose) console.log("[DEBUG] incIndex :: End; index:",index);

    if (onProcessElement())
        setTimeout(processElement, 2000 );
}

function getUrlVars() {
    var vars = {};
    var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi,    
    function(m,key,value) {
        vars[key] = value;
    });
    return vars;
}

function geUrlPageNum(){
    var result = getUrlVars()["page"];
    if (result && debugVerbose) console.log(result);
    else {
        if (debugVerbose) console.log("[DEBUG]param PAGE not found.");
        result = 1;
    }    
    return result;
}    

function timeMod( time1 ){
    var baseTime = time1 * setupDefaultDelay;
    var t = ((baseTime)*setupTimeModifier) + setupGlobalTimeOffset;
    if (t < baseTime*0.1) t = baseTime*0.1;
    if (t > baseTime*10) t = baseTime*10;
    return t;    
}

function reloadPage() {
    if (debugVerbose) console.log("[DEBUG] reloadPage :: Start; index:",index);
    scriptPaused = true;

    if (pageNum >= debugStopAfterPage) {
        return;
    }    


    var newUrl = "";
    newUrl = pageURL + "&limit=" + itemsPerPage + "&page=" + pageNum.toString();

    if (debugVerbose) console.log("[DEBUG] reloadPage :: ",newUrl);
    document.location.href = newUrl;

    setTimeout( mixamoScript, 2000 );
}

function onFinish(){
/*
    timeEnd = Date.now();
    scriptPaused = true;
    if (debugVerbose) console.log("[DEBUG]MixamoScript: Added (", numAssetsAdded, ")" );

    if (setupBeepOnFinish) {
        soundOnFinish();    
        setTimeout(promptOnEnd, 2000);
    } else setTimeout(promptOnEnd, 0);
*/    
}

if (!audioCtx) var audioCtx = new (window.AudioContext || window.webkitAudioContext || window.audioContext);
function beep(duration, frequency, volume, type, callback) {
    if (!setupPlaySounds)  return;

    var oscillator = audioCtx.createOscillator();
    var gainNode = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    if (volume){gainNode.gain.value = volume;};
    if (frequency){oscillator.frequency.value = frequency;}
    if (type){oscillator.type = type;}
    if (callback){oscillator.onended = callback;}

    oscillator.start();
    setTimeout(function(){oscillator.stop()}, (duration ? duration : 500));
}

function soundOnPageEnd(){

    setTimeout(function(){ beep(500, 500, 0.060, "triangle") }, 0);
    setTimeout(function(){ beep(500, 550, 0.035, "triangle") }, 600);
    setTimeout(function(){ beep(500, 600, 0.020, "triangle") }, 1200);
    setTimeout(function(){ beep(500, 650, 0.005, "triangle") }, 1800);
}

function beepOnAdd(){
    if (debugVerbose) console.log("[DEBUG] beepOnAdd :: Start; index:",index);
    beep(225, 200, 0.3, "triangle");
}    


/**/
function printToLogCurrItems(){

}

function onEndOfPage() {
    if (debugVerbose) console.log("[DEBUG] onEndOfPage :: Start; index:",index);
    scriptPaused = true;

    var t = 0;
    if (setupBeepOnPage) {
        t = 1000;
        setTimeout(soundOnPageEnd, 0);        
    }    

    //printToLogCurrItems();
    if (pageNum >= debugStopAfterPage) {
        setTimeout( onFinish, 2000 );
        if (debugVerbose) console.log("[DEBUG] onEndOfPage_Assets :: Start; pageNum:",pageNum,"; debugStopAfterPage:",debugStopAfterPage);
        return;
    } else {
        if (!debugNoPageTurn) {
            index = 0;
            ++pageNum;
            setTimeout(reloadPage, TimeToReloadPage);
        }
    }        

}

function isDownloadInProgress(){
    var elements = document.getElementsByClassName( "progress-bar" );
    if (elements && elements.length>0)
    {
        console.log("[DEBUG] Download In Progress");
        return true;
    } else {
        console.log("[DEBUG] Download Not In Progress");
        return false;
    }    
}

function onProcessElement() {
    if (debugVerbose) console.log("[DEBUG] onProcessElement :: start; index:",index);

    if (!items || items.length == 0) {
        if (debugVerbose) console.log("[DEBUG] onProcessElement :: items; index:",index);
        items = document.getElementsByClassName( setupElementID_Products );
    }    
    if (!button) {
        if (debugVerbose) console.log("[DEBUG] onProcessElement :: button; index:",index);
        //button = document.querySelector( setupElementID_AddToAssets );        
        var buttons = document.getElementsByClassName( setupElementID_Download );
        if (buttons && buttons.length>0)
            button = buttons[0];
    }    
    if (items.length != 0) {
        // Next Page
        if(index >= items.length) {
            // end of page
            onEndOfPage();
            return true;
        }
        items[index].click();    

        if (debugVerbose) console.log("[DEBUG] onProcessElement :: Clicked");
        return true;
    }

    return false;
}

function getAnimName(){
    if (debugVerbose) console.log("[DEBUG] getAnimName :: Start; index:",index);
    var element = document.getElementsByClassName( "text-center h5" );
    if (element && element.length>0) {
        if (debugVerbose) console.log("[DEBUG] getAnimName :: name:",element[0].textContent,"; Start; index:",index);
        return element[0].textContent;
    }    
    if (debugVerbose) console.log("[DEBUG] getAnimName :: Failed; Start; index:",index);
    return "[]";
}

function confirmDownload(){
    if (debugVerbose) console.log("[DEBUG] confirmDownload :: start; index:",index);
    var buttons = document.getElementsByClassName( setupElementID_DownloadConfirm );
    if (buttons && buttons.length>1)    {
        buttons[1].click();

        if (debugVerbose) console.log("[DEBUG] confirmDownload :: Button Clicked");

        setTimeout( processElementEx, 1000 );
    } else if (debugVerbose) console.log("[DEBUG] confirmDownload :: Button Not Found");
}

function initAnimDownload() {
    if (debugVerbose) console.log("[DEBUG] initAnimDownload :: Start; Index:",index);

    var currAnim = getAnimName();
    if (debugVerbose) console.log("[DEBUG] initAnimDownload :: Anim: ",currAnim,"; index:",index);

    // main button is ready to add asset)
    if (button.textContent == setupElementID_DownloadCaption) {
        if (buttonClicked == 0) {
            button.click();            

            //++buttonClicked;
            //++numAssetsAddedCurrPage;
            if (setupBeepOnAdd) beepOnAdd();            

            setTimeout( confirmDownload, 1000);

            if (debugVerbose) console.log("[DEBUG] initAnimDownload :: Added To Assets; buttonClicked:",buttonClicked);
            return true;
        }
    } else { // main button caption is something unexpected
        if (debugVerbose) console.log("[DEBUG] initAnimDownload :: button.textContent:",button.textContent);        
        return false;

    }
    if (debugVerbose) console.log("[DEBUG] initAnimDownload :: end");
}

function checkElement(elementName, desiredState) {
    var element = document.getElementsByName(elementName);
    if (element.length != 0) {
        if (element[0].checked != desiredState) {
            if (debugVerbose) console.log("[DEBUG] checkElement :: Clicked; previous state:",element[0].checked,"; elementName:",elementName,"; desiredState:",desiredState);
            element[0].click();
        } else if (debugVerbose) console.log("[DEBUG] checkElement :: Element already has desired state; elementName:",elementName,"; desiredState:",desiredState);

        return true;
    }
    if (debugVerbose) console.log("[DEBUG] checkElement :: Element not found; elementName:",elementName,"; desiredState:",desiredState);

    return false;
}        

function isElementChecked(elementName, desiredState) {
    var element = document.getElementsByName(elementName);
    if (element.length != 0) {
        if (debugVerbose) console.log("[DEBUG] isElementChecked :: elementName: ",elementName,"; desiredState: ",desiredState,"; current state:",element[0].checked,"; items found: ",element.length);
        if (element[0].checked != desiredState) {
            if (debugVerbose) console.log("[DEBUG] isElementChecked :: return: 0");
            return 0;
        } else {
            if (debugVerbose) console.log("[DEBUG] isElementChecked :: return: 1");
            return 1;
        }    
    }
    if (debugVerbose) console.log("[DEBUG] isElementChecked :: Element not found; elementName:",elementName,"; desiredState:",desiredState);
    return -1;
}

var waitingForDownloadToComplete = false;
function processElement() {    
    if (debugVerbose) console.log("[DEBUG] processElement :: start; index:",index);

    if (isDownloadInProgress()) {
    // give page time to prepare current download
        waitingForDownloadToComplete = true;
        setTimeout(processElement, 500 );
        return;
    }

    if (waitingForDownloadToComplete === true){
    // some extra time to complete downloading
        waitingForDownloadToComplete = false;
        setTimeout(processElement, 1000 );
        return;    
    }

    if (debugMaxIndex > 0 && NumOfFilesProcessed >= debugMaxIndex) {
        if (debugVerbose) console.log("[DEBUG] processElement :: debugMaxIndex:",debugMaxIndex,"; onFinish");
        onFinish();
        return;
    }        

    if ( scriptPaused ) return;

    retryCounter = 0;

    setTimeout(processElementEx, timeMod( 1 ));

    if (debugVerbose) console.log("[DEBUG] processElement :: End");
}

function processElementEx(){
    if (debugVerbose) {
        console.log("[DEBUG] processElementEx :: DownloadPending_Regular: ",DownloadPending_Regular);
        console.log("[DEBUG] processElementEx :: DownloadPending_Mirror: ",DownloadPending_Mirror);
        console.log("[DEBUG] processElementEx :: DownloadPending_InPlace: ",DownloadPending_InPlace);
        console.log("[DEBUG] processElementEx :: DownloadPending_InPlaceMirror: ",DownloadPending_InPlaceMirror);
    }    

    if (!isAnimPack())    {
        // download variants
        if (DownloadPending_Regular) {        
            if (initAnimDownload_Regular() ) {
                setTimeout(initAnimDownload, timeMod( 2 ) );                
                if (debugVerbose) console.log("[DEBUG] processElementEx :: call initAnimDownload");
            } else {
                setTimeout(processElementEx, timeMod( 1 ) );            
                if (debugVerbose) console.log("[DEBUG] processElementEx :: call processElementEx");
            }    
            DownloadPending_Regular = false;
            return;
        } else if (DownloadPending_Mirror){
            setTimeout(initAnimDownload_Mirror, timeMod( 1 ) );
            if (debugVerbose) console.log("[DEBUG] processElementEx :: call initAnimDownload_Mirror");
            return;
        } else if (DownloadPending_InPlace){
            setTimeout(initAnimDownload_InPlace, timeMod( 1 ) );
            if (debugVerbose) console.log("[DEBUG] processElementEx :: call initAnimDownload_InPlace");
            return;
        } else if (DownloadPending_InPlaceMirror){
            setTimeout(initAnimDownload_InPlaceMirror, timeMod( 1 ) );
            if (debugVerbose) console.log("[DEBUG] processElementEx :: call initAnimDownload_InPlaceMirror");
            return;
        }        
    } else {
        if (debugVerbose) console.log("[DEBUG] processElementEx :: skipping Pack");
    }

    if (debugVerbose) {
        console.log("[DEBUG] processElementEx :: DownloadPending_Regular: ",DownloadPending_Regular);
        console.log("[DEBUG] processElementEx :: DownloadPending_Mirror: ",DownloadPending_Mirror);
        console.log("[DEBUG] processElementEx :: DownloadPending_InPlace: ",DownloadPending_InPlace);
        console.log("[DEBUG] processElementEx :: DownloadPending_InPlaceMirror: ",DownloadPending_InPlaceMirror);
    }    

    //DownloadPending_Any                = false;
    setTimeout(incIndex, timeMod(2) );    
}

function initAnimDownload_Regular(){
    if (debugVerbose) console.log("[DEBUG] initAnimDownload_Regular :: Start");
    r1 = isElementChecked(setupElementID_Mirror, false);
    r2 = isElementChecked(setupElementID_InPlace, false);

    if (debugVerbose) console.log("[DEBUG] initAnimDownload_Regular :: R1: ",r1,", R2: ",r2);

    if ((download_RootExclude===true && r2>-1) || (download_RootExclussive===true && r2<0)) {
        if (debugVerbose) console.log("[DEBUG] initAnimDownload_Regular :: download_RootExclude: ",download_RootExclude);
        if (debugVerbose) console.log("[DEBUG] initAnimDownload_Regular :: download_RootExclussive: ",download_RootExclussive);
        return false;
    }
    if (download_RootMirrored===true) {
        DownloadPending_Mirror = true;
        return false;
    } else {        
        return true;
    }        
}

function initAnimDownload_Mirror(){
    r1 = isElementChecked(setupElementID_Mirror, true);
    r2 = isElementChecked(setupElementID_InPlace, false);

    if (debugVerbose) console.log("[DEBUG] initAnimDownload_Mirror :: R1: ",r1,", R2: ",r2);

    if (r1 === -1) {
        // option n/a
        DownloadPending_Mirror = false;
        setTimeout(processElementEx, timeMod( 1 ) );
    } else if (r1===0 || r2===0) {
        // req options not yet not checked        
        if (r1===0) checkElement(setupElementID_Mirror, true);        
        if (r2===0) checkElement(setupElementID_InPlace, false);
        setTimeout(processElementEx, timeMod( 1 ) );
    } else {
        // req met
        setTimeout(initAnimDownload, timeMod( 2 ) );
        DownloadPending_Mirror = false;
    }
}

function initAnimDownload_InPlace(){
    //r1 = isElementChecked(setupElementID_InPlace, true);
    r1 = isElementChecked(setupElementID_InPlace, false);
    r2 = isElementChecked(setupElementID_Mirror, false);

    if (debugVerbose) console.log("[DEBUG] initAnimDownload_InPlace :: R1: ",r1,", R2: ",r2);

    if (r1 === -1) {
        // option n/a
        DownloadPending_InPlace = false;
        setTimeout(processElementEx, timeMod( 1 ) );
    } else if (r1===0 || r2===0) {
        // req options not yet not checked
        //if (r1===0) checkElement(setupElementID_InPlace, true);
        if (r1===0) checkElement(setupElementID_InPlace, false);
        if (r2===0) checkElement(setupElementID_Mirror, false);
        setTimeout(processElementEx, timeMod( 1 ) );
    } else {
        // req met
        setTimeout(initAnimDownload, timeMod( 2 ) );
        DownloadPending_InPlace = false;
    }
}

function initAnimDownload_InPlaceMirror(){
    r1 = isElementChecked(setupElementID_InPlace, true);
    r2 = isElementChecked(setupElementID_Mirror, true);

    if (debugVerbose) console.log("[DEBUG] initAnimDownload_InPlaceMirror :: R1: ",r1,", R2: ",r2);

    if (r1 === -1 || r2 === -1) {
        // option n/a
        DownloadPending_InPlaceMirror = false;
        setTimeout(processElementEx, timeMod( 1 ) );
    } else if (r1===0 || r2===0) {
        // req options not yet not checked
        if (r1===0) checkElement(setupElementID_InPlace, true);
        if (r2===0) checkElement(setupElementID_Mirror, true);
        setTimeout(processElementEx, timeMod( 1 ) );
    } else {
        // req met
        setTimeout(initAnimDownload, timeMod( 2 ) );
        DownloadPending_InPlaceMirror = false;
    }
}

function mixamoScript(){
    timeStart = Date.now();

    if (debugVerbose) console.log("[DEBUG] mixamoScript :: Start; index:",index);
    var pagination = document.querySelector(setupElementID_Pagination);
    if (pagination) pageMax       = parseInt(pagination.textContent, 10);
    else pageMax       = 1;

    scriptPaused       = false;
    items  = document.getElementsByClassName( setupElementID_Products );

    index                         = -1;

    // get products or retry later
    if (items.length == 0) {
        ++debugFailedStart;
        if (debugFailedStart > 2) {
            reloadPage();
            return;
        }    
        setTimeout(mixamoScript, 2000 );
        return;
    }
    debugFailedStart = 0;

    // process first element    
    incIndex();
    //processElement();    
}    

if (Setup_DownloadAllExceptInPlace===true)                     DownloadAllExceptInPlace();
else if (Setup_DownloadAllExceptInPlace_Mirrored===true)     DownloadAllExceptInPlace_Mirrored();
else if (Setup_DownloadOnlyInPlace===true)                     DownloadOnlyInPlace();
else if (Setup_DownloadOnlyInPlace_Mirrored===true)         DownloadOnlyInPlace_Mirrored();
else if (Setup_DownloadOnlyRoot===true)                     DownloadOnlyRoot();
else if (Setup_DownloadOnlyRoot_Mirrored===true)             DownloadOnlyRoot_Mirrored();

mixamoScript();