import { useLocation, Link } from 'react-router-dom'
import { 
  Home, 
  MapPin, 
  Maximize2, 
  BedDouble, 
  Bath, 
  CheckCircle,
  TrendingUp,
  Building2
} from 'lucide-react'

export default function ResultsPage() {
  const location = useLocation()
  const { prediction, propertyData, userRole, transactionType } = location.state || {}

  if (!prediction) {
    return (
      <div className="text-center py-16">
        <p className="text-gray-600 mb-4">No prediction data available</p>
        <Link to="/predict" className="text-primary-600 hover:underline">
          Go to Prediction Page
        </Link>
      </div>
    )
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('fr-TN', {
      style: 'currency',
      currency: 'TND',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  const formatNumber = (num) => {
    return new Intl.NumberFormat('fr-TN').format(num)
  }

  // Calculate price per sqm
  const pricePerSqm = propertyData.surface > 0 
    ? prediction.predicted_price / propertyData.surface 
    : 0

  // Count amenities
  const amenities = [
    { key: 'has_piscine', label: 'Piscine', icon: '🏊' },
    { key: 'has_garage', label: 'Garage', icon: '🚗' },
    { key: 'has_jardin', label: 'Jardin', icon: '🌳' },
    { key: 'has_terrasse', label: 'Terrasse', icon: '🏡' },
    { key: 'has_ascenseur', label: 'Ascenseur', icon: '🛗' },
    { key: 'is_meuble', label: 'Meublé', icon: '🛋️' },
    { key: 'has_chauffage', label: 'Chauffage', icon: '🔥' },
    { key: 'has_climatisation', label: 'Climatisation', icon: '❄️' },
  ].filter(amenity => propertyData[amenity.key])

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
          <CheckCircle className="w-10 h-10 text-green-600" />
        </div>
        <h1 className="text-3xl font-bold mb-2">Prediction Complete!</h1>
        <p className="text-gray-600">
          Here's our AI-powered price estimate for your property
        </p>
      </div>

      {/* Main Prediction Card */}
      <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200">
        <div className="text-center mb-6">
          <div className="inline-block px-4 py-1 bg-blue-600 text-white rounded-full text-sm font-medium mb-4">
            {transactionType === 'rent' ? '📅 Monthly Rent' : '🏠 Sale Price'}
          </div>
          <div className="text-5xl font-bold text-blue-900 mb-2">
            {formatPrice(prediction.predicted_price)}
          </div>
          <div className="text-sm text-gray-600">
            Currency: {prediction.currency || 'TND'}
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Model: {prediction.model}
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mt-6 pt-6 border-t border-blue-200">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-700">
              {formatNumber(pricePerSqm.toFixed(0))} TND
            </div>
            <div className="text-sm text-gray-600">Price per m²</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-700">
              {propertyData.surface} m²
            </div>
            <div className="text-sm text-gray-600">Total Surface</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-700">
              {propertyData.price_segment || 'Medium'}
            </div>
            <div className="text-sm text-gray-600">Market Segment</div>
          </div>
        </div>
      </div>

      {/* Property Details */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Location & Type */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Building2 className="w-5 h-5 mr-2 text-blue-600" />
            Property Information
          </h3>
          <div className="space-y-3">
            <div className="flex items-start">
              <MapPin className="w-5 h-5 mr-3 text-gray-400 mt-0.5" />
              <div>
                <div className="font-medium">{propertyData.region}</div>
                <div className="text-sm text-gray-600">{propertyData.city}</div>
              </div>
            </div>
            <div className="flex items-center">
              <Home className="w-5 h-5 mr-3 text-gray-400" />
              <div>
                <span className="font-medium capitalize">{propertyData.property_type}</span>
              </div>
            </div>
            <div className="flex items-center">
              <Maximize2 className="w-5 h-5 mr-3 text-gray-400" />
              <div>
                <span className="font-medium">{propertyData.surface} m²</span>
              </div>
            </div>
          </div>
        </div>

        {/* Rooms & Features */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Rooms & Layout</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center p-3 bg-gray-50 rounded-lg">
              <BedDouble className="w-5 h-5 mr-3 text-gray-600" />
              <div>
                <div className="text-2xl font-bold">{propertyData.rooms || 0}</div>
                <div className="text-sm text-gray-600">Rooms</div>
              </div>
            </div>
            <div className="flex items-center p-3 bg-gray-50 rounded-lg">
              <Bath className="w-5 h-5 mr-3 text-gray-600" />
              <div>
                <div className="text-2xl font-bold">{propertyData.bathrooms || 0}</div>
                <div className="text-sm text-gray-600">Bathrooms</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Amenities */}
      {amenities.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Property Features</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {amenities.map((amenity) => (
              <div
                key={amenity.key}
                className="flex items-center p-3 bg-green-50 border border-green-200 rounded-lg"
              >
                <span className="text-2xl mr-2">{amenity.icon}</span>
                <span className="text-sm font-medium text-green-900">{amenity.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Market Context */}
      <div className="card bg-amber-50 border-2 border-amber-200">
        <div className="flex items-start">
          <TrendingUp className="w-6 h-6 mr-3 text-amber-600 mt-1" />
          <div>
            <h3 className="text-lg font-semibold mb-2">Market Context</h3>
            <p className="text-gray-700 text-sm leading-relaxed">
              This price prediction is based on our AI model trained on thousands of similar properties 
              in Tunisia. The {transactionType === 'rent' ? 'rent' : 'sale'} price reflects current market 
              conditions in <strong>{propertyData.region}</strong> for <strong>{propertyData.property_type}</strong> properties.
              {transactionType === 'rent' && ' This is the estimated monthly rent.'}
              {transactionType === 'sale' && ' This is the estimated sale price.'}
            </p>
          </div>
        </div>
      </div>

      {/* Similar Properties from KNN */}
      {prediction.similar_properties && prediction.similar_properties.length > 0 && (
        <div className="card">
          <h3 className="text-2xl font-bold mb-4 flex items-center">
            <Home className="w-6 h-6 mr-2 text-blue-600" />
            Similar Properties in Our Dataset
          </h3>
          <p className="text-gray-600 mb-6 text-sm">
            Here are the 5 most similar properties from our training data, based on size, rooms, and bathrooms.
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {prediction.similar_properties.map((property, index) => (
              <div
                key={index}
                className="border-2 border-gray-200 rounded-lg p-4 hover:border-blue-400 transition-colors bg-white"
              >
                {/* Similarity Badge */}
                <div className="flex justify-between items-start mb-3">
                  <div className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                    {(property.similarity_score * 100).toFixed(0)}% Match
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-blue-600">
                      {formatPrice(property.price)}
                    </div>
                  </div>
                </div>

                {/* Location */}
                <div className="flex items-start mb-2">
                  <MapPin className="w-4 h-4 mr-2 text-gray-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <div className="font-medium">{property.region}</div>
                    <div className="text-gray-600 text-xs">{property.city}</div>
                  </div>
                </div>

                {/* Property Type */}
                <div className="flex items-center mb-2">
                  <Building2 className="w-4 h-4 mr-2 text-gray-400" />
                  <span className="text-sm capitalize">{property.property_type}</span>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-gray-200 text-xs">
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-1">
                      <Maximize2 className="w-3 h-3 text-gray-500" />
                    </div>
                    <div className="font-semibold">{property.surface} m²</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-1">
                      <BedDouble className="w-3 h-3 text-gray-500" />
                    </div>
                    <div className="font-semibold">{property.rooms} rooms</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-1">
                      <Bath className="w-3 h-3 text-gray-500" />
                    </div>
                    <div className="font-semibold">{property.bathrooms} bath</div>
                  </div>
                </div>

                {/* Amenities */}
                {(property.has_piscine || property.has_garage || property.has_jardin || 
                  property.has_terrasse || property.has_ascenseur) && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="flex flex-wrap gap-1">
                      {property.has_piscine && (
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">🏊 Pool</span>
                      )}
                      {property.has_garage && (
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">🚗 Garage</span>
                      )}
                      {property.has_jardin && (
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">🌳 Garden</span>
                      )}
                      {property.has_terrasse && (
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">🏡 Terrace</span>
                      )}
                      {property.has_ascenseur && (
                        <span className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded">🛗 Elevator</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-col sm:flex-row justify-center gap-4">
        <Link 
          to="/predict" 
          className="btn btn-primary px-8 py-3 text-center"
        >
          New Prediction
        </Link>
        <Link 
          to="/" 
          className="btn btn-secondary px-8 py-3 text-center"
        >
          Back to Home
        </Link>
      </div>
    </div>
  )
}
