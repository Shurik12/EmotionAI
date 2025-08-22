import React, { useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';

const ApplicationModal = ({ selectedPlan, closeModal }) => {
	const [formData, setFormData] = useState({
		name: '',
		phone: '',
		company: ''
	});
	const [isSubmitting, setIsSubmitting] = useState(false);
	const { language } = useLanguage();

	const handleInputChange = (e) => {
		const { name, value } = e.target;
		setFormData(prev => ({
			...prev,
			[name]: value
		}));
	};

	const handleSubmit = async (e) => {
		e.preventDefault();
		setIsSubmitting(true);

		try {
			const submissionData = {
				plan: selectedPlan,
				...formData
			};

			// In a real app, you would use fetch API here
			console.log('Form submission:', submissionData);

			// Simulate API call
			await new Promise(resolve => setTimeout(resolve, 1000));

			alert(language === 'ru'
				? 'Спасибо за вашу заявку! Мы свяжемся с вами в ближайшее время.'
				: 'Thank you for your application! We will contact you soon.'
			);

			closeModal();
			setFormData({ name: '', phone: '', company: '' });
		} catch (error) {
			console.error('Error submitting form:', error);
			alert(language === 'ru'
				? 'Произошла ошибка при отправке заявки. Пожалуйста, попробуйте позже.'
				: 'An error occurred while submitting the application. Please try again later.'
			);
		} finally {
			setIsSubmitting(false);
		}
	};

	const t = (key, replacements = {}) => {
		return translate(key, language, replacements);
	};
	
	return (
		<div id="applicationModal" className="modal" style={{
			display: 'block',
			position: 'fixed',
			zIndex: 1000,
			left: 0,
			top: 0,
			width: '100%',
			height: '100%',
			backgroundColor: 'rgba(0, 0, 0, 0.5)'
		}}>
			<div className="modal-content" style={{
				backgroundColor: '#fefefe',
				margin: '15% auto',
				padding: '20px',
				border: '1px solid #888',
				width: '80%',
				maxWidth: '500px',
				borderRadius: '8px',
				position: 'relative'
			}}>
				<span
					className="close-modal"
					onClick={closeModal}
					style={{
						color: '#aaa',
						float: 'right',
						fontSize: '28px',
						fontWeight: 'bold',
						cursor: 'pointer',
						position: 'absolute',
						right: '15px',
						top: '10px'
					}}
				>
					&times;
				</span>

				<h2>{t('application_title')}</h2>

				<form id="planApplicationForm" onSubmit={handleSubmit}>
					<input
						type="hidden"
						id="selectedPlan"
						name="plan"
						value={selectedPlan}
					/>

					<div className="form-group" style={{ marginBottom: '15px' }}>
						<label htmlFor="name" style={{ display: 'block', marginBottom: '5px' }}>
							{t('form_name')}
						</label>
						<input
							type="text"
							id="name"
							name="name"
							value={formData.name}
							onChange={handleInputChange}
							required
							style={{
								width: '100%',
								padding: '8px',
								border: '1px solid #ddd',
								borderRadius: '4px'
							}}
						/>
					</div>

					<div className="form-group" style={{ marginBottom: '15px' }}>
						<label htmlFor="phone" style={{ display: 'block', marginBottom: '5px' }}>
							{t('form_phone')}
						</label>
						<input
							type="tel"
							id="phone"
							name="phone"
							value={formData.phone}
							onChange={handleInputChange}
							required
							style={{
								width: '100%',
								padding: '8px',
								border: '1px solid #ddd',
								borderRadius: '4px'
							}}
						/>
					</div>

					<div className="form-group" style={{ marginBottom: '15px' }}>
						<label htmlFor="company" style={{ display: 'block', marginBottom: '5px' }}>
							{t('form_company')}
						</label>
						<input
							type="text"
							id="company"
							name="company"
							value={formData.company}
							onChange={handleInputChange}
							style={{
								width: '100%',
								padding: '8px',
								border: '1px solid #ddd',
								borderRadius: '4px'
							}}
						/>
					</div>

					<button
						type="submit"
						className="btn"
						disabled={isSubmitting}
						style={{
							backgroundColor: isSubmitting ? '#ccc' : 'var(--primary-color)',
							color: 'white',
							border: 'none',
							padding: '10px 20px',
							borderRadius: '4px',
							cursor: isSubmitting ? 'not-allowed' : 'pointer',
							width: '100%'
						}}
					>
						{isSubmitting ?
							(language === 'ru' ? 'Отправка...' : 'Submitting...') :
							t('form_submit')
						}
					</button>
				</form>
			</div>
		</div>
	);
};

export default ApplicationModal;