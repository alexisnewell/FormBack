
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

export default api;
