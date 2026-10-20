"use strict";
const $ = id => document.getElementById(id);
const fieldIds = ["full_name","document_number","date_of_birth","expiry_date"];
let detections=[], selectedIndex=0, sourceImage=null, sourceObjectUrl=null, edits=[];
function setStatus(message,error=false){$("status").textContent=message;$("status").className=error?"error":"";}
function saveEdits(){if(!edits[selectedIndex])return;for(const id of fieldIds)edits[selectedIndex][id]=$(id).value;}
function draw(){
 if(!sourceImage)return;const canvas=$("preview");canvas.width=sourceImage.naturalWidth;canvas.height=sourceImage.naturalHeight;
 const ctx=canvas.getContext("2d");ctx.drawImage(sourceImage,0,0);
 detections.forEach((d,i)=>{const [x1,y1,x2,y2]=d.box;ctx.strokeStyle=i===selectedIndex?"#00e070":"#ffae00";ctx.lineWidth=Math.max(3,canvas.width/350);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.fillStyle=ctx.strokeStyle;ctx.font=`${Math.max(16,canvas.width/45)}px sans-serif`;ctx.fillText(`Card ${i+1}`,x1+3,Math.max(20,y1-5));});
}
function showSelected(){
 const d=detections[selectedIndex];if(!d)return;draw();$("crop").src=d.crop;
 $("portrait").hidden=!d.portrait;$("no-portrait").hidden=!!d.portrait;
 if(d.portrait)$("portrait").src=d.portrait;else $("portrait").removeAttribute("src");
 $("confidence").textContent=d.confidence==null?"Detection score unavailable":`Detection confidence: ${(d.confidence*100).toFixed(1)}%`;
 $("ocr-confidence").textContent=d.ocr_mean_confidence==null?"OCR confidence unavailable":`Mean Tesseract word confidence: ${d.ocr_mean_confidence}%`;
 $("ocr-text").textContent=d.ocr_text||"(No text recognized)";
 for(const id of fieldIds)$(id).value=edits[selectedIndex][id]||"";
}
function record(){saveEdits();const d=detections[selectedIndex];return {fields:{...edits[selectedIndex]},detection:{class_name:d.class_name,confidence:d.confidence,box_xyxy:d.box},raw_ocr_text:d.ocr_text,ocr_mean_confidence:d.ocr_mean_confidence};}
function download(name,contents,mime){const url=URL.createObjectURL(new Blob([contents],{type:mime}));const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csvCell(v){let s=String(v??"");if(/^\s*[=+\-@]/.test(s))s="'"+s;return `"${s.replaceAll('"','""')}"`;}
$("upload-form").addEventListener("submit",async e=>{
 e.preventDefault();const file=$("file").files[0];if(!file)return;
 $("results").hidden=true;detections=[];edits=[];setStatus("Running hosted card detection and local OCR…");
 const form=new FormData();form.append("file",file);
 try{
  const response=await fetch("/api/extract",{method:"POST",body:form,cache:"no-store"});
  const payload=await response.json();if(!response.ok)throw new Error(payload.detail||`HTTP ${response.status}`);
  if(!Array.isArray(payload.detections))throw new Error("Unexpected server response");
  if(!payload.detections.length){setStatus("No card detected. Try a clear, front-facing photo.");return;}
  if(sourceObjectUrl)URL.revokeObjectURL(sourceObjectUrl);sourceObjectUrl=URL.createObjectURL(file);
  sourceImage=new Image();await new Promise((resolve,reject)=>{sourceImage.onload=resolve;sourceImage.onerror=reject;sourceImage.src=sourceObjectUrl;});
  detections=payload.detections;edits=detections.map(d=>({...d.fields}));selectedIndex=0;
  const choice=$("card-choice");choice.replaceChildren();detections.forEach((_,i)=>choice.add(new Option(`Card ${i+1}`,String(i))));
  $("results").hidden=false;showSelected();setStatus(`Detected ${detections.length} card(s). Review before exporting.`);
 }catch(err){setStatus(err.message||"Extraction failed",true);}
});
$("card-choice").addEventListener("change",e=>{saveEdits();selectedIndex=Number(e.target.value);showSelected();});
$("json-export").addEventListener("click",()=>download("idvision.json",JSON.stringify(record(),null,2),"application/json"));
$("csv-export").addEventListener("click",()=>{const r=record();const keys=[...fieldIds,"raw_ocr_text"];const row=[...fieldIds.map(k=>r.fields[k]),r.raw_ocr_text];download("idvision.csv",keys.map(csvCell).join(",")+"\r\n"+row.map(csvCell).join(",")+"\r\n","text/csv;charset=utf-8");});
window.addEventListener("pagehide",()=>{if(sourceObjectUrl)URL.revokeObjectURL(sourceObjectUrl);});
