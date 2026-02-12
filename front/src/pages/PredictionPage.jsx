import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import PropertyForm from '../components/PropertyForm'
import api from '../services/api'

export default function PredictionPage() {
  const [userRole, setUserRole] = useState('seller')
  const [transactionType, setTransactionType] = useState('sale')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  const handleSubmit = async (propertyData) => {
    setIsLoading(true)
    setError(null)

    try {
      // Prepare features for rent or sale model
      const features = {
        region: propertyData.region,
        city: propertyData.city || propertyData.region, // Fallback to region if city is empty
        property_type: propertyData.property_type,
        price_segment: propertyData.price_segment || 'Medium',
        surface: parseFloat(propertyData.surface),
        rooms: parseInt(propertyData.rooms) || 0,
        bathrooms: parseInt(propertyData.bathrooms) || 0,
        has_piscine: propertyData.has_piscine || false,
        has_garage: propertyData.has_garage || false,
        has_jardin: propertyData.has_jardin || false,
        has_terrasse: propertyData.has_terrasse || false,
        has_ascenseur: propertyData.has_ascenseur || false,
        is_meuble: propertyData.is_meuble || false,
        has_chauffage: propertyData.has_chauffage || false,
        has_climatisation: propertyData.has_climatisation || false,
      }

      // Add property_type_cluster for sale model
      if (transactionType === 'sale') {
        features.property_type_cluster = 0 // Default cluster
      }

      const requestData = {
        features,
      }

      // Call the appropriate endpoint based on transaction type
      let response
      if (transactionType === 'rent') {
        response = await api.predictRent(requestData)
      } else {
        response = await api.predictSale(requestData)
      }
      
      // Navigate to results page with data
      navigate('/results', {
        state: {
          prediction: response.data,
          propertyData: features,
          userRole,
          transactionType,
        },
      })
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction')
      console.error('Prediction error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="card mb-8">
        <h1 className="text-3xl font-bold mb-6">Property Price Prediction</h1>

        {/* Role Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">I am a:</label>
          <div className="flex space-x-4">
            <button
              onClick={() => setUserRole('seller')}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                userRole === 'seller'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Seller/Landlord
            </button>
            <button
              onClick={() => setUserRole('buyer')}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                userRole === 'buyer'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Buyer/Renter
            </button>
          </div>
        </div>

        {/* Transaction Type */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Transaction Type:</label>
          <div className="flex space-x-4">
            <button
              onClick={() => setTransactionType('sale')}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                transactionType === 'sale'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Sale
            </button>
            <button
              onClick={() => setTransactionType('rent')}
              className={`flex-1 py-3 rounded-lg font-medium transition-colors ${
                transactionType === 'rent'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Rent
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}
      </div>

      <PropertyForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        userRole={userRole}
      />
    </div>
  )
}
