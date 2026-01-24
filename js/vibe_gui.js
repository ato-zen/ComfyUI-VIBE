import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("%c🎨 VIBE Extension: Active", "color: cyan; font-weight: bold;");

app.registerExtension({
    name: "ato-zen.VIBE.Download",
    async nodeCreated(node) {
        if (node.comfyClass === "VIBE_Editor") {
            
            // Default state
            const btn = node.addWidget("button", "📥 Check / Download Model", null, () => {
                
                // Prevent double clicking if already working
                if (btn.disabled) return;

                if (confirm("Check for VIBE weights updates or download if missing?")) {
                    btn.name = "⏳ Checking...";
                    btn.disabled = true;

                    api.fetchApi("/vibe/download_model", { method: "POST" })
                       .then(response => response.json())
                       .then(data => {
                           if (data.status === "error") {
                               btn.name = "❌ Server Error";
                               btn.disabled = false;
                               alert(data.message);
                           }
                           // If status is 'started', we wait for socket events below
                       })
                       .catch(err => {
                           btn.name = "❌ Connection Error";
                           btn.disabled = false;
                           alert("API Error: " + err);
                       });
                }
            });

            // Listen for detailed status updates from Python
            const onVibeStatus = (event) => {
                const status = event.detail.status;
                const msg = event.detail.message;

                if (status === "progress") {
                    // Python is actively downloading/updating
                    btn.name = msg; // e.g. "Downloading..." or "Updating..."
                } 
                else if (status === "finished") {
                    btn.name = "✅ Ready";
                    btn.disabled = false;
                    alert(msg);
                } 
                else if (status === "error") {
                    btn.name = "❌ Failed";
                    btn.disabled = false;
                    alert(msg);
                }
            };

            api.addEventListener("vibe_status", onVibeStatus);
        }
    }
});
