import { render, screen } from '@testing-library/react';
import { ValenceArousal } from './DetectorResults';

describe('ValenceArousal', () => {
  it('renders nothing when features is null', () => {
    const { container } = render(<ValenceArousal features={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders nothing when features is empty', () => {
    const { container } = render(<ValenceArousal features={{}} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders title and feature value', () => {
    render(<ValenceArousal features={{ valence: 0.75 }} />);
    expect(screen.getByText('Valence')).toBeInTheDocument();
    expect(screen.getByText('75.0%')).toBeInTheDocument();
  });

  it('renders multiple features', () => {
    render(<ValenceArousal features={{ valence: 0.6, arousal: 0.3 }} />);
    expect(screen.getByText('Valence')).toBeInTheDocument();
    expect(screen.getByText('Arousal')).toBeInTheDocument();
    expect(screen.getByText('60.0%')).toBeInTheDocument();
    expect(screen.getByText('30.0%')).toBeInTheDocument();
  });

  it('renders 0% correctly', () => {
    render(<ValenceArousal features={{ valence: 0 }} />);
    expect(screen.getByText('0.0%')).toBeInTheDocument();
  });

  it('renders bars with correct width', () => {
    const { container } = render(<ValenceArousal features={{ valence: 0.42 }} />);
    const fill = container.querySelector('.emotion-fill');
    expect(fill).toHaveStyle({ width: '42.0%' });
  });

  it('renders bars with correct feature color', () => {
    const { container } = render(<ValenceArousal features={{ valence: 0.5 }} />);
    const fill = container.querySelector('.emotion-fill');
    // valence color from constants is #E91E63
    expect(fill).toHaveStyle({ backgroundColor: '#E91E63' });
  });
});
