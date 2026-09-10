import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import api from '../services/api';

interface User {
  email: string;
  id: number;
  is_active: boolean;
  is_admin: boolean;
  node_id: string | null;
  wallet_id: string | null;
}

interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (response: LoginResponse) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  isAuthenticated: boolean;
  isLoading: boolean;
  hasWallet: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const clearAuth = useCallback(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('refreshToken');
    setToken(null);
    setUser(null);
    setIsLoading(false);
  }, []);

  // Listen for auth:logout events from the API interceptor
  useEffect(() => {
    const handleLogout = () => {
      clearAuth();
    };

    window.addEventListener('auth:logout', handleLogout);
    return () => window.removeEventListener('auth:logout', handleLogout);
  }, [clearAuth]);

  useEffect(() => {
    const storedToken = localStorage.getItem('token');
    if (storedToken) {
      setToken(storedToken);
      fetchUser().catch(() => {
        // Error already handled in fetchUser (token cleared, user set to null)
      });
    } else {
      setIsLoading(false);
    }
  }, []);

  const fetchUser = async () => {
    setIsLoading(true);
    try {
      // Use the api instance which handles token refresh automatically
      const response = await api.get('/api/users/me');
      setUser(response.data);
    } catch (error) {
      console.error('Failed to fetch user', error);
      clearAuth();
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (response: LoginResponse) => {
    const { access_token, refresh_token } = response;
    
    // Store both tokens
    localStorage.setItem('token', access_token);
    localStorage.setItem('refreshToken', refresh_token);
    
    setToken(access_token);
    await fetchUser();
  };

  const refreshUser = async () => {
    if (!token) return;
    try {
      const response = await api.get('/api/users/me');
      setUser(response.data);
    } catch (error) {
      console.error('Failed to refresh user', error);
      // Don't clear token on refresh failure, just log it
    }
  };

  const logout = async () => {
    // Try to revoke the refresh token on the server
    const refreshToken = localStorage.getItem('refreshToken');
    if (refreshToken) {
      try {
        await api.post('/api/auth/logout', { refresh_token: refreshToken });
      } catch (error) {
        // Ignore errors during logout - we'll clear locally anyway
        console.error('Logout API call failed', error);
      }
    }
    
    clearAuth();
  };

  const hasWallet = !!(user?.node_id && user?.wallet_id);

  return (
    <AuthContext.Provider value={{ 
      user, 
      token, 
      login, 
      logout, 
      refreshUser, 
      isAuthenticated: !!user, 
      isLoading,
      hasWallet 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
