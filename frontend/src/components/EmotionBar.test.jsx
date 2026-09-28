import { render, screen } from '@testing-library/react';
import { EmotionBar } from './EmotionBar';

describe('EmotionBar', () => {
  it('renders the emotion name and percentage', () => {
    render(<EmotionBar emotion="anger" probability={0.5} />);
    expect(screen.getByText('Anger')).toBeInTheDocument();
    expect(screen.getByText('50.0%')).toBeInTheDocument();
  });

  it('renders 0% correctly', () => {
    render(<EmotionBar emotion="neutral" probability={0} />);
    expect(screen.getByText('Neutral')).toBeInTheDocument();
    expect(screen.getByText('0.0%')).toBeInTheDocument();
  });

  it('renders 100% correctly', () => {
    render(<EmotionBar emotion="happiness" probability={1} />);
    expect(screen.getByText('Happiness')).toBeInTheDocument();
    expect(screen.getByText('100.0%')).toBeInTheDocument();
  });

  it('handles string probability values', () => {
    render(<EmotionBar emotion="sadness" probability="0.75" />);
    expect(screen.getByText('Sadness')).toBeInTheDocument();
    expect(screen.getByText('75.0%')).toBeInTheDocument();
  });

  it('renders the bar with the correct width', () => {
    const { container } = render(<EmotionBar emotion="joy" probability={0.42} />);
    const fill = container.querySelector('.emotion-fill');
    expect(fill).toHaveStyle({ width: '42.0%' });
  });

  it('renders with the correct emotion color', () => {
    const { container } = render(<EmotionBar emotion="anger" probability={0.5} />);
    const fill = container.querySelector('.emotion-fill');
    // anger color from constants is #ea4335
    expect(fill).toHaveStyle({ backgroundColor: '#ea4335' });
  });

  it('falls back to raw key if translation is missing', () => {
    render(<EmotionBar emotion="unknown_emotion" probability={0.1} />);
    expect(screen.getByText('emotions.unknown_emotion')).toBeInTheDocument();
  });
});
