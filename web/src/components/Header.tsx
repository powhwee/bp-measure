import { Link } from 'react-router-dom';

export function Header() {
    return (
        <div className="navbar bg-base-100 shadow-sm">
            <div className="navbar-start">
                <div className="flex flex-col">
                    <Link to="/" className="btn btn-ghost text-xl font-bold h-auto min-h-0 py-1">Measure Blood Pressure</Link>
                    <span className="text-xs pl-4 opacity-70 leading-none">Experimental, not for medical advice</span>
                </div>
            </div>
            <div className="navbar-end hidden lg:flex">
                <ul className="menu menu-horizontal px-1 font-medium">
                    <li><Link to="/">Home</Link></li>
                    <li><Link to="/measure">Measure</Link></li>
                    <li><Link to="/history">History</Link></li>
                    <li><Link to="/calibrate">Calibration</Link></li>
                </ul>
            </div>
            <div className="navbar-end lg:hidden">
                <div className="dropdown dropdown-end">
                    <div tabIndex={0} role="button" className="btn btn-ghost btn-circle">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h7" /></svg>
                    </div>
                    <ul tabIndex={0} className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52">
                        <li><Link to="/">Home</Link></li>
                        <li><Link to="/measure">Measure</Link></li>
                        <li><Link to="/history">History</Link></li>
                        <li><Link to="/calibrate">Calibration</Link></li>
                    </ul>
                </div>
            </div>
        </div>
    );
}
