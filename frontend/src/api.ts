
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 30000,
});

export interface UploadResponse {
  video_id: string;
  filename: string;
  size_bytes: number;
  fps: number;
  frame_count: number;
  duration_s: number;
}

export const uploadVideo = async (
  file: File
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post(
    "/videos/upload",
    formData
  );

  return response.data;
};

export const startAnalysis = async (
  videoId: string
) => {
  const response = await api.post(
    `/videos/${videoId}/analyze`
  );

  return response.data;
};

export const getAnalysisStatus = async (
  videoId: string
) => {
  const response = await api.get(
    `/videos/${videoId}/status`
  );

  return response.data;
};

export const getAnalysisResults = async (
  videoId: string
) => {
  const response = await api.get(
    `/videos/${videoId}/results`
  );

  return response.data;
};

export type ReportFormat = "pdf" | "csv";

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Downloads the report and hands it to the browser as a file.
export const downloadReport = async (
  videoId: string,
  format: ReportFormat
): Promise<void> => {
  try {
    const response = await api.get(`/videos/${videoId}/report`, {
      params: { format },
      responseType: "blob",
    });

    saveBlob(response.data, `formback-report-${videoId.slice(0, 8)}.${format}`);
  } catch (err) {
    // With responseType "blob" an error body also arrives as a Blob. Turn it
    // back into JSON so callers can read `err.response.data.detail` as usual.
    if (axios.isAxiosError(err) && err.response?.data instanceof Blob) {
      try {
        err.response.data = JSON.parse(await err.response.data.text());
      } catch {
        /* not JSON; leave it */
      }
    }
    throw err;
  }
};

export default api;
