import axios from 'axios';

// API configuration
// Use relative URLs so it works both in development (with proxy) and Docker (with nginx proxy)
export const API_URL = '';

// Setup axios interceptors for global error handling
export const setupAxiosInterceptors = () => {
  axios.interceptors.response.use(
    (response) => response,
    (error) => {
      // Handle "Method Not Allowed" (405) - redirect to home
      if (error.response?.status === 405) {
        console.warn('Method Not Allowed - redirecting to home');
        window.location.href = '/';
        return Promise.reject(error);
      }

      // Handle "Method Not Allowed" in response body
      if (error.response?.data?.detail === 'Method Not Allowed') {
        console.warn('Method Not Allowed in response - redirecting to home');
        window.location.href = '/';
        return Promise.reject(error);
      }

      // Let other errors pass through to be handled by individual components
      return Promise.reject(error);
    }
  );
};
