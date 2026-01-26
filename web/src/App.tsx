import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';

import { Home } from './pages/Home';
import { Measure } from './pages/Measure';
import { History } from './pages/History';
import { Calibrate } from './pages/Calibrate';

function App() {
  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen bg-base-100 font-sans">
        <Header />
        <main className="flex-grow container mx-auto px-4 py-8 max-w-6xl">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/measure" element={<Measure />} />
            <Route path="/history" element={<History />} />
            <Route path="/calibrate" element={<Calibrate />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
