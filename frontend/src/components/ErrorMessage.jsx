import React from 'react';

export const ErrorMessage = ({ message }) => {
  if (!message) return null;

  return (
    <div className="error-message">
      <span className="error-icon">⚠️</span>
      <span className="error-text">{message}</span>
    </div>
  );
};